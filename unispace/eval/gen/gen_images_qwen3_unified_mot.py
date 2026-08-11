# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3Unified + Qwen3 MoT 图像生成脚本。

与 gen_images_qwen3_mot.py 相比的主要区别：
  - vae_type='qwen3_unified'，使用 RAEWrapper 代替 Flux VAE
  - latent_patch_size=1（RAE downsample=16 已完成空间降采样）
  - use_qwen3_unified=True
  - decode: latent shape [1, 1280, h, w]，不需要 einsum reshape
  - norm: RAEWrapper.decode 输出已经是 [-1,1]，直接 * 0.5 + 0.5 即可
"""

import os
import json
import argparse
import time
import random
import gc

import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image

from transformers import AutoTokenizer

from modeling.unimm.unimm_mot import UnimmMoT, UnimmConfig
from modeling.unimm.qwen3_mot import Qwen3MoTForConditionalGeneration
from modeling.unimm.qwen3_vl_mot import Qwen3VLMoTForConditionalGeneration
from modeling.unimm.qwen3 import NaiveCache
from modeling.qwen3.configuration_qwen3 import Qwen3Config
from modeling.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from modeling.autoencoder_rae import load_rae

import torch._dynamo
torch._dynamo.config.disable = True


from eval.gen.gen_images_qwen3_mot import MEIGEN_TEST_PROMPTS  # noqa: E402


def get_device_info():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
        backend = "hccl"
        device_count = torch.npu.device_count()
        def set_device(idx): torch.npu.set_device(idx)
        def synchronize(): torch.npu.synchronize()
        def empty_cache(): torch.npu.empty_cache()
    elif torch.cuda.is_available():
        device_type = "cuda"
        backend = "nccl"
        device_count = torch.cuda.device_count()
        def set_device(idx): torch.cuda.set_device(idx)
        def synchronize(): torch.cuda.synchronize()
        def empty_cache(): torch.cuda.empty_cache()
    else:
        raise RuntimeError("No CUDA or NPU device available")
    return {
        'type': device_type, 'backend': backend, 'device_count': device_count,
        'set_device': set_device, 'synchronize': synchronize, 'empty_cache': empty_cache,
    }


def move_generation_input_to_device(generation_input, device):
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def decode_latent(latent, h, w, latent_ch, vae_model, use_merge_gen, device_type):
    if use_merge_gen:
        from modeling.unimm.unimm_mot import UnimmMoT
        latent = UnimmMoT.spatial_unshuffle(latent, h, w)
    latent = latent.reshape(1, h, w, latent_ch).permute(0, 3, 1, 2)
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        image = vae_model.decode(latent)
    return ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()


def generate_image(
    prompt,
    num_timesteps=50,
    cfg_scale=4.0,
    cfg_interval=None,
    cfg_renorm_min=0.9,
    timestep_shift=1.0,
    max_t=1.0,
    num_images=1,
    resolution=256,
    inference_mode='flash',
    device=None,
    device_type='cuda',
    gen_model=None,
    tokenizer=None,
    new_token_ids=None,
    vae_model=None,
    seed=None,
    intermediates_dir=None,
):
    if cfg_interval is None:
        cfg_interval = [0, 1.0]
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

    _llm_cfg = gen_model.config.llm_config
    num_layers = (_llm_cfg.text_config.num_hidden_layers
                  if hasattr(_llm_cfg, 'text_config') and _llm_cfg.text_config is not _llm_cfg
                  else _llm_cfg.num_hidden_layers)
    past_key_values = NaiveCache(num_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    # 1. encode prompt prefix into KV cache
    generation_input_text, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_text = move_generation_input_to_device(generation_input_text, device)
    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input_text)

    # 2. prepare VAE latent slots
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
        new_token_ids=new_token_ids,
        generator=generator,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    # 3. CFG prefix (text-free)
    cfg_past_key_values = NaiveCache(num_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images
    cfg_prompt_input, cfg_newlens, cfg_new_rope = gen_model.prepare_cfg_prompts(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        num_images=num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    cfg_prompt_input = move_generation_input_to_device(cfg_prompt_input, device)
    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
        cfg_past_key_values = gen_model.forward_cache_update_text(cfg_past_key_values, **cfg_prompt_input)

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    # 4. flow matching denoising
    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
        result = gen_model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            max_t=max_t,
            cfg_text_scale=cfg_scale,
            cfg_renorm_min=cfg_renorm_min,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            cfg_text_past_key_values=cfg_past_key_values,
            cfg_text_packed_position_ids=generation_input_cfg['cfg_packed_position_ids'],
            cfg_text_key_values_lens=generation_input_cfg['cfg_key_values_lens'],
            cfg_text_packed_query_indexes=generation_input_cfg['cfg_packed_query_indexes'],
            cfg_text_packed_key_value_indexes=generation_input_cfg['cfg_packed_key_value_indexes'],
            generation_input_text=generation_input_text,
            inference_mode=inference_mode,
            return_intermediates=(intermediates_dir is not None),
            **generation_input,
        )
        if intermediates_dir is not None:
            unpacked_latent, intermediate_latents = result
        else:
            unpacked_latent = result
            intermediate_latents = []

    # 5. VAE decode
    latent_ch = gen_model.config.vae_config.z_channels  # 1280
    latent_ds = gen_model.config.vae_config.downsample  # 16
    h = w = resolution // latent_ds  # 256//16 = 16
    use_merge_gen = gen_model.config.use_spatial_merge_gen

    # save intermediate steps
    if intermediates_dir is not None and intermediate_latents:
        os.makedirs(intermediates_dir, exist_ok=True)
        # only decode first image in batch for speed
        split_sizes = generation_input['packed_seqlens'].tolist()
        split_sizes = [s - 2 for s in split_sizes]  # strip bos/eos
        for step_idx, t_val, packed_xt in intermediate_latents:
            per_img = packed_xt.split(split_sizes)
            latent = per_img[0].float()
            arr = decode_latent(latent, h, w, latent_ch, vae_model, use_merge_gen, device_type)
            Image.fromarray(arr).save(
                os.path.join(intermediates_dir, f"step_{step_idx:03d}_t{t_val:.3f}.png"))

    image_list = []
    for latent in unpacked_latent:
        arr = decode_latent(latent.float(), h, w, latent_ch, vae_model, use_merge_gen, device_type)
        image_list.append(Image.fromarray(arr))

    return image_list


def load_model(args, device, device_type='cuda'):
    llm_type = getattr(args, 'llm_type', 'qwen3')
    is_qwen3vl = (llm_type == 'qwen3vl')

    if is_qwen3vl:
        llm_config = Qwen3VLConfig.from_pretrained(args.llm_path)
        llm_config.text_config.qk_norm = True
        llm_config.text_config.tie_word_embeddings = False
        llm_config.text_config._attn_implementation = 'sdpa'
        llm_config._attn_implementation = 'sdpa'
    else:
        llm_config = Qwen3Config.from_pretrained(args.llm_path)
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config._attn_implementation = 'sdpa'

    print("  [1/4] Loading RAE (Qwen3Unified)...")
    vae_model, vae_config = load_rae(
        local_path=args.vae_path,
        rae_accelerate_root=getattr(args, 'rae_accelerate_root', None),
    )

    config = UnimmConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=1,
        max_latent_size=args.max_latent_size,
        use_qwen_vit=False,
        use_moe=True,
        use_qwen3_unified=True,
        timestep_shift=1.0,
        share_unified2llm=getattr(args, 'share_unified2llm', True),
        use_spatial_merge=getattr(args, 'use_spatial_merge', False),
        use_spatial_merge_gen=getattr(args, 'use_spatial_merge_gen', None),
        use_spatial_merge_und=getattr(args, 'use_spatial_merge_und', None),
        pred_type=getattr(args, 'pred_type', 'v_pred'),
        use_mrope=is_qwen3vl,
    )

    if is_qwen3vl:
        print("  [2/4] Instantiating Qwen3VLMoTForConditionalGeneration...")
        language_model = Qwen3VLMoTForConditionalGeneration.from_pretrained(
            args.llm_path, config=llm_config)
    else:
        print("  [2/4] Instantiating Qwen3MoTForConditionalGeneration...")
        language_model = Qwen3MoTForConditionalGeneration.from_pretrained(
            args.llm_path, config=llm_config,
            key_mapping=Qwen3MoTForConditionalGeneration._checkpoint_conversion_mapping)

    print("  [3/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    print("  [4/4] Building UnimmMoT and loading checkpoint...")
    model = UnimmMoT(language_model, None, tokenizer, config)

    new_token_ids = {
        'bos_token_id': tokenizer.encode("<|im_start|>")[0],
        'eos_token_id': tokenizer.encode("<|im_end|>")[0],
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        'start_of_image': tokenizer.encode("<|vision_start|>")[0],
        'end_of_image': tokenizer.encode("<|vision_end|>")[0],
    }

    model_state_dict_path = os.path.join(args.model_path, "model.safetensors")
    if os.path.exists(model_state_dict_path):
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        # Detect ExactMergerConnector format: checkpoint has unified2llm_und.norm.weight
        # but model was initialized with MLPconnector (different shape).
        # strict=False does NOT skip shape mismatches — must rebuild before load_state_dict.
        # Same logic as vlmeval_adapter_qwen3vl_unified_mot.py.
        if 'unified2llm_und.norm.weight' in model_state_dict and hasattr(model, 'unified2llm_und'):
            fc1_w = model_state_dict['unified2llm_und.fc1.weight']
            fc2_w = model_state_dict['unified2llm_und.fc2.weight']
            nm_w  = model_state_dict['unified2llm_und.norm.weight']
            model.rebuild_unified2llm_und_as_exact_merger(
                total_in=fc1_w.shape[1],
                merger_hidden=fc1_w.shape[0],
                out_dim=fc2_w.shape[0],
                sem_per_patch=nm_w.shape[0],
            )
            print(f"  Rebuilt unified2llm_und as ExactMergerConnector "
                  f"({fc1_w.shape[1]}→{fc1_w.shape[0]}→{fc2_w.shape[0]})", flush=True)
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(f"  Checkpoint: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys}")
        del model_state_dict
    else:
        print(f"  WARNING: model.safetensors not found in {args.model_path}")
    gc.collect()

    model = model.to(device).to(dtype=torch.bfloat16).eval()
    # vae_model 保持 float32（RAE 内部 latent_mean/var 是 float32）
    vae_model = vae_model.to(device).eval()

    return model, tokenizer, new_token_ids, vae_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--use_meigen_test_cases", action="store_true")
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument("--use_spatial_merge", action="store_true", default=False)
    parser.add_argument("--use_spatial_merge_gen", type=lambda x: x.lower() != 'false', default=None)
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != 'false', default=None)
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="JSONL with h5_img_path/h5_img_root fields; orig image saved as orig.png per prompt")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--llm_type", type=str, default="qwen3",
                        help="LLM backbone type: 'qwen3' (text-only) or 'qwen3vl' (Qwen3-VL multimodal)")
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--rae-accelerate-root", type=str,
                        default=None)
    parser.add_argument("--inference-mode", type=str, default="flash")
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mrope_section", type=str, default="8,28,28")
    parser.add_argument("--timestep-shift", type=float, default=1,
                        help="Inference timestep shift. 1/s=1/sqrt(z*h*w/base)=1/sqrt(1280*16*16/4096)~0.112, concentrates steps near noise end (t=0).")
    parser.add_argument("--pred_type", type=str, default="v_pred",
                        help="Prediction type used at training time: 'v_pred' (default) or 'x_pred'.")
    parser.add_argument("--max_t", type=float, default=1.0,
                        help="Max timestep for denoising schedule. <1.0 avoids x_pred singularity near t=1.")
    parser.add_argument("--save_steps", action="store_true", default=False,
                        help="Save decoded image at every denoising step (only first prompt, first image).")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    device_info = get_device_info()
    device_type = device_info['type']

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.distributed:
        import datetime, torch.distributed as dist
        dist.init_process_group(backend=device_info['backend'], timeout=datetime.timedelta(seconds=3600))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device_info['set_device'](local_rank)
        device = torch.device(f"{device_type}:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device_info['set_device'](0)
        device = torch.device(f"{device_type}:0")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3Unified MOT model...")
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    print("Model loaded.")

    prompts = []
    metadatas = []
    if args.use_meigen_test_cases:
        prompts = MEIGEN_TEST_PROMPTS
        print(f"Using {len(prompts)} MeiGen test cases")
    elif args.prompt is not None:
        prompts = [args.prompt]
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if args.metadata_file is not None:
        import json as _json
        with open(args.metadata_file, "r") as f:
            metadatas = [_json.loads(l) for l in f if l.strip()]
        if not prompts:
            prompts = []
            for m in metadatas:
                cap = m.get("generated_caption", "{}")
                try:
                    prompts.append(_json.loads(cap)["en"]["概述"])
                except Exception:
                    prompts.append("")

    if world_size > 1:
        prompts_per_gpu = (len(prompts) + world_size - 1) // world_size
        start_idx = rank * prompts_per_gpu
        end_idx = min(start_idx + prompts_per_gpu, len(prompts))
        local_prompts = list(enumerate(prompts))[start_idx:end_idx]
    else:
        local_prompts = list(enumerate(prompts))

    cfg_scale = getattr(args, 'cfg_scale', 4.0)
    num_timesteps = getattr(args, 'inference_steps', 50)
    num_images_per_prompt = getattr(args, 'num_images_per_prompt', 1)
    batch_size = min(getattr(args, 'batch_size', 4), num_images_per_prompt)

    for prompt_idx, prompt in local_prompts:
        print(f"[{prompt_idx + 1}/{len(prompts)}] {prompt[:80]}")
        safe_prompt = prompt.replace(" ", "_").replace("'", "").replace('"', '').replace("/", "_")[:50]
        prompt_output_dir = os.path.join(args.output_dir, f"prompt_{prompt_idx:03d}_{safe_prompt}")
        os.makedirs(prompt_output_dir, exist_ok=True)
        with open(os.path.join(prompt_output_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # 保存原始训练图片（如果有 metadata）
        if metadatas and prompt_idx < len(metadatas):
            try:
                import h5py
                meta = metadatas[prompt_idx]
                h5_root = meta.get("h5_img_root", "")
                h5_path = meta.get("h5_img_path", "")
                h5_file, h5_idx = h5_path.split("#")
                full_h5 = os.path.join(h5_root, h5_file)
                import io
                with h5py.File(full_h5, "r") as hf:
                    img_bytes = hf["images"][int(h5_idx)][:]
                orig_img = Image.open(io.BytesIO(img_bytes.tobytes())).convert("RGB").resize((args.resolution, args.resolution))
                orig_img.save(os.path.join(prompt_output_dir, "orig.png"))
            except Exception as e:
                print(f"  Warning: failed to save orig image: {e}")

        all_images = []
        for batch_idx in range(0, num_images_per_prompt, batch_size):
            current_batch_size = min(batch_size, num_images_per_prompt - batch_idx)
            try:
                save_steps = getattr(args, 'save_steps', False)
                intermediates_dir = (
                    os.path.join(prompt_output_dir, "steps")
                    if save_steps and batch_idx == 0
                    else None
                )
                image_list = generate_image(
                    prompt=prompt,
                    cfg_scale=cfg_scale,
                    cfg_renorm_min=0.9,
                    cfg_interval=[0, 1.0],
                    timestep_shift=getattr(args, 'timestep_shift', 1),
                    max_t=getattr(args, 'max_t', 1.0),
                    intermediates_dir=intermediates_dir,
                    num_timesteps=num_timesteps,
                    num_images=current_batch_size,
                    resolution=args.resolution,
                    inference_mode=getattr(args, 'inference_mode', 'flash'),
                    device=device,
                    device_type=device_type,
                    gen_model=gen_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    vae_model=vae_model,
                    seed=args.seed,
                )
                all_images.extend(image_list)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        for img_idx, image in enumerate(all_images):
            image_path = os.path.join(prompt_output_dir, f"image_{img_idx:03d}.png")
            image.save(image_path)
        print(f"  Saved {len(all_images)} images")
        device_info['empty_cache']()

    if args.distributed:
        import torch.distributed as dist
        dist.barrier()


if __name__ == "__main__":
    main()
