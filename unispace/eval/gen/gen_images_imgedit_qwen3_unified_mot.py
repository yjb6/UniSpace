"""
Qwen3Unified (RAE) 图像编辑推理脚本。

基于 gen_images_imgedit.py，适配 qwen3_unified_mot 架构：
  - ref 图: RAE und 支路编码 (替换 SigLIP ViT + VAE@t=1)
  - vae_type: qwen3_unified (RAE)
  - latent_patch_size: 1
  - VAE decode: 简单 reshape（无 einsum）

序列结构:
  conditional:    [ref_und] [user\\n{instruction}] [assistant\\n] → [target denoise]
  unconditional:  [ref_und]                        [assistant\\n] → [target denoise]

用法:
  torchrun --nproc_per_node=8 eval/gen/gen_images_imgedit_qwen3_unified_mot.py \\
      --output_dir ./output --metadata_file edit_prompts.jsonl \\
      --model-path /path/to/stage2_edit_ckpt \\
      --llm-path /path/to/Qwen3-8B \\
      --vae-path /path/to/rae_config.yaml \\
      --rae-accelerate-root /path/to/RAE_accelerate

metadata_file 格式（JSONL，每行一条）:
  {"id": "001", "image_path": "/path/to/src.jpg", "prompt": "Make the sky purple"}
"""

import os
import json
import argparse
import time
import random
import gc

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from transformers import AutoTokenizer

from modeling.unimm.unimm_mot import UnimmMoT, UnimmConfig
from modeling.unimm.qwen3_mot import Qwen3MoTForConditionalGeneration
from modeling.unimm.qwen3_vl_mot import Qwen3VLMoTForConditionalGeneration
from modeling.unimm.qwen3 import NaiveCache
from modeling.qwen3.configuration_qwen3 import Qwen3Config
from modeling.unimm.qwen3_vl import Qwen3VLConfig
from modeling.autoencoder_rae import load_rae
from data.data_utils import pil_img2rgb, get_flattened_position_ids_extrapolate

import torch._dynamo
torch._dynamo.config.disable = True


# ═══════════════════════════════════════════════════════
# 设备检测
# ═══════════════════════════════════════════════════════

def get_device_info():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
        backend = "hccl"
        def set_device(idx): torch.npu.set_device(idx)
        def synchronize(): torch.npu.synchronize()
        def empty_cache(): torch.npu.empty_cache()
    elif torch.cuda.is_available():
        device_type = "cuda"
        backend = "nccl"
        def set_device(idx): torch.cuda.set_device(idx)
        def synchronize(): torch.cuda.synchronize()
        def empty_cache(): torch.cuda.empty_cache()
    else:
        raise RuntimeError("No CUDA or NPU device available")
    return {
        'type': device_type, 'backend': backend,
        'set_device': set_device, 'synchronize': synchronize, 'empty_cache': empty_cache,
    }


def move_to_device(d, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


# ═══════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════

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
        visual_und=True,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=args.max_latent_size,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=1,
        max_latent_size=args.max_latent_size,
        use_qwen_vit=False,
        use_moe=True,
        use_qwen3_unified=True,
        share_unified2llm=args.share_unified2llm,
        use_spatial_merge=False,
        use_spatial_merge_gen=False,
        use_spatial_merge_und=args.use_spatial_merge_und,
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
            key_mapping=Qwen3MoTForConditionalGeneration._checkpoint_conversion_mapping,
        )

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
        file_size_gb = os.path.getsize(model_state_dict_path) / 1024**3
        print(f"  Loading checkpoint: {model_state_dict_path}", flush=True)
        print(f"  File size: {file_size_gb:.2f} GB", flush=True)
        t0 = time.time()
        state_dict = load_file(model_state_dict_path, device="cpu")
        print(f"  load_file done in {time.time()-t0:.1f}s, keys={len(state_dict)}", flush=True)
        state_dict.pop('latent_pos_embed.pos_embed', None)
        state_dict.pop('vit_pos_embed.pos_embed', None)
        # Detect ExactMergerConnector format: checkpoint has unified2llm_und.norm.weight
        # but model was initialized with MLPconnector (different shape).
        # strict=False does NOT skip shape mismatches — must rebuild before load_state_dict.
        if 'unified2llm_und.norm.weight' in state_dict and hasattr(model, 'unified2llm_und'):
            fc1_w = state_dict['unified2llm_und.fc1.weight']
            fc2_w = state_dict['unified2llm_und.fc2.weight']
            nm_w  = state_dict['unified2llm_und.norm.weight']
            model.rebuild_unified2llm_und_as_exact_merger(
                total_in=fc1_w.shape[1],
                merger_hidden=fc1_w.shape[0],
                out_dim=fc2_w.shape[0],
                sem_per_patch=nm_w.shape[0],
            )
            print(f"  Rebuilt unified2llm_und as ExactMergerConnector "
                  f"({fc1_w.shape[1]}→{fc1_w.shape[0]}→{fc2_w.shape[0]})", flush=True)
        t1 = time.time()
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"  load_state_dict done in {time.time()-t1:.1f}s", flush=True)
        print(f"  Checkpoint: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}", flush=True)
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys}", flush=True)
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys}", flush=True)
        del state_dict
    else:
        print(f"  WARNING: model.safetensors not found in {args.model_path}", flush=True)
    gc.collect()

    print("  Moving model to device...", flush=True)
    t2 = time.time()
    model = model.to(device).to(dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device).eval()
    print(f"  Model on device in {time.time()-t2:.1f}s", flush=True)

    return model, tokenizer, new_token_ids, vae_model


# ═══════════════════════════════════════════════════════
# ref 图 und 支路编码
# ═══════════════════════════════════════════════════════

def encode_ref_image(pil_image, vae_model, mot_model, image_size, device, device_type,
                     ref_max_size=None, ref_min_size=256, ref_max_pixels=1_048_576,
                     ref_use_mar=False, ref_mar_resolution=1024):
    """PIL → RAE und 支路编码 → (merged_tokens, pos_ids)

    三种模式（互斥，优先级：ref_use_mar > ref_max_size > 默认）：

    默认（ref_max_size=None, ref_use_mar=False）:
        强制 resize 到正方形 (image_size, image_size)。

    ref_max_size=N:
        保持宽高比，MaxLongEdgeMinShortEdgeResize(max_size=N, min_size=ref_min_size,
        stride=32, max_pixels=ref_max_pixels)，与训练 ref_vit_image_transform_args 一致。

    ref_use_mar=True:
        snap 到最近的 mar_{ref_mar_resolution} bucket（与训练 editref config 一致），
        ref 和 target 使用同一套 bucket，分辨率完全对齐。
    """
    import torchvision.transforms.functional as TF
    from data.transforms import MaxLongEdgeMinShortEdgeResize

    img = pil_image.convert('RGB')

    if ref_use_mar:
        # mar bucket 模式：snap 到最近的 mar_{ref_mar_resolution} bucket
        target_h, target_w = compute_target_size_from_ref(pil_image, ref_mar_resolution)
        img = img.resize((target_w, target_h), Image.BICUBIC)  # PIL: (width, height)
        h_img, w_img = target_h, target_w
    elif ref_max_size is not None:
        # 可变模式：保持宽高比，与训练 ref_vit_image_transform_args 一致
        resizer = MaxLongEdgeMinShortEdgeResize(
            max_size=ref_max_size,
            min_size=ref_min_size,
            stride=32,
            max_pixels=ref_max_pixels,
        )
        img = resizer(img)
        w_img, h_img = img.size  # PIL: (width, height)
    else:
        # 原有行为：强制正方形
        img = img.resize((image_size, image_size), Image.BICUBIC)
        h_img = w_img = image_size

    img_tensor = TF.to_tensor(img) * 2.0 - 1.0   # [3, H, W] in [-1,1]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
        z = vae_model.encode(img_tensor)  # [1, z_ch, h, w]

    h_grid = h_img // 16
    w_grid = w_img // 16
    z_tokens = z[0].permute(1, 2, 0).reshape(h_grid * w_grid, -1).to(dtype=torch.bfloat16)

    if mot_model.config.use_spatial_merge_und:
        merged_tokens = UnimmMoT.spatial_merge(z_tokens, h_grid, w_grid)
        patch_size = 32  # 16 * 2
    else:
        merged_tokens = z_tokens
        patch_size = 16

    pos_ids = get_flattened_position_ids_extrapolate(
        h_img, w_img,
        patch_size=patch_size,
        max_num_patches_per_side=mot_model.max_latent_size,
    ).to(device)

    return merged_tokens, pos_ids


def prepare_unified_vit_input(curr_kvlens, curr_rope, vit_tokens, pos_ids, new_token_ids, device):
    """打包 forward_cache_update_unified_vit 所需输入（参考 vlmeval_adapter_qwen3_unified_mot.py）"""
    packed_text_ids, packed_text_indexes = [], []
    packed_vit_token_indexes = []
    packed_position_ids_list, packed_seqlens, packed_indexes = [], [], []
    packed_key_value_indexes = []

    _curr = curr = 0
    newlens, new_rope_out = [], []
    for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen

        packed_text_ids.append(new_token_ids['start_of_image'])
        packed_text_indexes.append(_curr)
        packed_indexes.append(curr)
        curr += 1
        _curr += 1

        num_img_tokens = vit_tokens.shape[0]
        packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
        packed_indexes.extend(range(curr, curr + num_img_tokens))
        curr += num_img_tokens
        _curr += num_img_tokens

        packed_text_ids.append(new_token_ids['end_of_image'])
        packed_text_indexes.append(_curr)
        packed_indexes.append(curr)
        curr += 1
        _curr += 1

        pos_1d = torch.full((num_img_tokens + 2,), curr_position_id, dtype=torch.long)
        packed_position_ids_list.append(pos_1d)
        packed_seqlens.append(num_img_tokens + 2)
        newlens.append(curr_kvlen + num_img_tokens + 2)
        new_rope_out.append(curr_position_id + num_img_tokens + 2)

    return {
        'packed_text_ids':          torch.tensor(packed_text_ids, dtype=torch.long).to(device),
        'packed_text_indexes':      torch.tensor(packed_text_indexes, dtype=torch.long).to(device),
        'packed_vit_tokens':        vit_tokens,
        'packed_vit_token_indexes': torch.tensor(packed_vit_token_indexes, dtype=torch.long).to(device),
        'packed_vit_position_ids':  pos_ids,
        'packed_position_ids':      torch.cat(packed_position_ids_list).to(device),
        'packed_seqlens':           torch.tensor(packed_seqlens, dtype=torch.int).to(device),
        'packed_indexes':           torch.tensor(packed_indexes, dtype=torch.long).to(device),
        'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long).to(device),
        'key_values_lens':          torch.tensor(curr_kvlens, dtype=torch.int).to(device),
    }, newlens, new_rope_out


# ═══════════════════════════════════════════════════════
# VAE decode
# ═══════════════════════════════════════════════════════

def decode_latent(latent, h, w, latent_ch, vae_model, use_spatial_merge_gen, device_type):
    if use_spatial_merge_gen:
        latent = UnimmMoT.spatial_unshuffle(latent, h, w)
    latent = latent.reshape(1, h, w, latent_ch).permute(0, 3, 1, 2)
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        image = vae_model.decode(latent)
    return ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def compute_target_size_from_ref(ref_image, resolution, stride=16):
    """按 ref 图宽高比 snap 到最近的 aspect-ratio bucket，与训练分布对齐。

    resolution=1024 → mar_1024，resolution=512 → mar_512，其余按比例计算。
    bucket key = H/W，value = [H, W]。
    """
    from data.data_utils import MULTI_RESOLUTION_MAP
    bucket_key = f'mar_{resolution}'
    if bucket_key in MULTI_RESOLUTION_MAP:
        bucket = MULTI_RESOLUTION_MAP[bucket_key]
        pil_w, pil_h = ref_image.size  # PIL: (width, height)
        ref_ratio = pil_h / pil_w      # H/W
        best_key = min(bucket.keys(), key=lambda k: abs(float(k) - ref_ratio))
        H, W = bucket[best_key]
        return (int(H), int(W))
    else:
        # 没有对应 bucket，按长边对齐 resolution
        pil_w, pil_h = ref_image.size
        if pil_w >= pil_h:
            target_w = resolution
            target_h = resolution * pil_h / pil_w
        else:
            target_h = resolution
            target_w = resolution * pil_w / pil_h
        target_h = max(stride, round(target_h / stride) * stride)
        target_w = max(stride, round(target_w / stride) * stride)
        return (int(target_h), int(target_w))


# ═══════════════════════════════════════════════════════
# 编辑推理核心
# ═══════════════════════════════════════════════════════

def editing_image(
    gen_model,
    tokenizer,
    new_token_ids,
    vae_model,
    ref_image,
    prompt,
    ref_image_size=448,
    ref_max_size=None,
    ref_min_size=256,
    ref_use_mar=False,
    ref_mar_resolution=1024,
    target_size=(256, 256),
    num_timesteps=50,
    cfg_text_scale=4.0,
    cfg_interval=None,
    cfg_renorm_min=0.0,
    timestep_shift=1.0,
    device=None,
    device_type='cuda',
    seed=None,
):
    if cfg_interval is None:
        cfg_interval = [0, 1.0]

    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    _llm_cfg = gen_model.config.llm_config
    num_layers = (_llm_cfg.text_config.num_hidden_layers
                  if hasattr(_llm_cfg, 'text_config') and _llm_cfg.text_config is not _llm_cfg
                  else _llm_cfg.num_hidden_layers)
    h, w = target_size

    # --- ref 图 und 编码（条件路径）---
    vit_tokens, pos_ids = encode_ref_image(
        ref_image, vae_model, gen_model, ref_image_size, device, device_type,
        ref_max_size=ref_max_size, ref_min_size=ref_min_size,
        ref_use_mar=ref_use_mar, ref_mar_resolution=ref_mar_resolution)

    # ========== conditional 路径 ==========
    past_key_values = NaiveCache(num_layers)
    newlens = [0]
    new_rope = [0]

    vit_inp, newlens, new_rope = prepare_unified_vit_input(
        newlens, new_rope, vit_tokens, pos_ids, new_token_ids, device)
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_unified_vit(past_key_values, **vit_inp)

    # user\n{instruction}
    inp_text, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens, curr_rope=new_rope,
        prompts=[f"user\n{prompt}"], tokenizer=tokenizer,
        new_token_ids=new_token_ids, raw=True,
    )
    inp_text = move_to_device(inp_text, device)
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **inp_text)

    # assistant\n
    inp_asst, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens, curr_rope=new_rope,
        prompts=["assistant\n"], tokenizer=tokenizer,
        new_token_ids=new_token_ids, raw=True,
    )
    inp_asst = move_to_device(inp_asst, device)
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **inp_asst)

    # 目标图 latent slot
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens, curr_rope=new_rope,
        image_sizes=[(h, w)], new_token_ids=new_token_ids, generator=generator,
    )
    generation_input = move_to_device(generation_input, device)

    # ========== unconditional 路径（只保留 ref 图，drop 指令）==========
    cfg_past_key_values = NaiveCache(num_layers)
    cfg_newlens = [0]
    cfg_new_rope = [0]

    vit_inp_cfg, cfg_newlens, cfg_new_rope = prepare_unified_vit_input(
        cfg_newlens, cfg_new_rope, vit_tokens, pos_ids, new_token_ids, device)
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        cfg_past_key_values = gen_model.forward_cache_update_unified_vit(cfg_past_key_values, **vit_inp_cfg)

    # 只有 assistant\n，跳过 user 指令
    inp_asst_cfg, cfg_newlens, cfg_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_newlens, curr_rope=cfg_new_rope,
        prompts=["assistant\n"], tokenizer=tokenizer,
        new_token_ids=new_token_ids, raw=True,
    )
    inp_asst_cfg = move_to_device(inp_asst_cfg, device)
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        cfg_past_key_values = gen_model.forward_cache_update_text(cfg_past_key_values, **inp_asst_cfg)

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens, curr_rope=cfg_new_rope, image_sizes=[(h, w)],
    )
    generation_input_cfg = move_to_device(generation_input_cfg, device)

    # ========== flow matching 生成 ==========
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            timestep_shift=timestep_shift,
            cfg_text_past_key_values=cfg_past_key_values,
            cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
            cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
            cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
            cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
            inference_mode='flash',
            **generation_input,
        )

    # ========== VAE decode ==========
    latent_ch = gen_model.config.vae_config.z_channels    # 1280
    latent_ds = gen_model.config.vae_config.downsample    # 16
    h_lat = h // latent_ds
    w_lat = w // latent_ds
    use_merge_gen = getattr(gen_model.config, 'use_spatial_merge_gen', False)
    arr = decode_latent(
        unpacked_latent[0].float(), h_lat, w_lat, latent_ch, vae_model, use_merge_gen, device_type)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Image editing with Qwen3Unified MOT model")
    # 输入
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="JSONL file, each line: {id, image_path, prompt}")
    parser.add_argument("--origin_img_root", type=str, default=None,
                        help="原图根目录，配合 singleturn.json 中 id 字段（相对路径）拼接完整路径")
    # 输出
    parser.add_argument("--output_dir", type=str, required=True)
    # 模型
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--llm_type", type=str, default="qwen3",
                        help="LLM 类型：qwen3（text-only）或 qwen3vl（含内置 VIT）")
    parser.add_argument("--vae-type", type=str, default="qwen3_unified")  # job_runner 固定传，忽略
    parser.add_argument("--rae-accelerate-root", type=str,
                        default=None)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != 'false', default=False,
                        help="False = diffmlp (独立 gen/und MLP)")
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != 'false', default=True,
                        help="True = und 侧 2x2 spatial merge（undmerge 系列）")
    # 推理参数
    parser.add_argument("--ref_image_size", type=int, default=448,
                        help="ref 图强制正方形编码分辨率（ref_max_size 未设置时生效）")
    parser.add_argument("--ref-max-size", type=int, default=None,
                        help="可变 ref 模式：长边上限，保持宽高比（与训练 ref_vit_image_transform 一致）")
    parser.add_argument("--ref-min-size", type=int, default=256,
                        help="可变 ref 模式：短边下限（默认 256，对应训练配置）")
    parser.add_argument("--ref-use-mar", action="store_true",
                        help="mar bucket 模式：ref snap 到最近的 mar_{ref_mar_resolution} bucket，与 editref 训练完全一致")
    parser.add_argument("--ref-mar-resolution", type=int, default=1024,
                        help="mar bucket 模式下的目标分辨率（默认 1024 → mar_1024）")
    parser.add_argument("--resolution", type=int, default=256,
                        help="目标图生成分辨率（正方形边长）")
    parser.add_argument("--match-ref-size", action="store_true",
                        help="按 ref 图宽高比生成，总像素≈resolution²，stride=16 对齐")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # 分布式
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    device_info = get_device_info()
    device_type = device_info['type']

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.distributed:
        import datetime
        import torch.distributed as dist
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

    if rank == 0:
        print("Loading Qwen3Unified MOT edit model...")
    t0 = time.time()
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    if rank == 0:
        print(f"Model loaded in {time.time() - t0:.1f}s")

    # --- 准备任务列表 ---
    tasks = []
    if args.image_path and args.prompt:
        tasks = [{'id': 'single', 'image_path': args.image_path, 'prompt': args.prompt}]
    elif args.metadata_file:
        with open(args.metadata_file, "r") as f:
            content = f.read().strip()
            if content.startswith('['):
                tasks = json.loads(content)
            elif content.startswith('{') and not content.startswith('{"'):
                data = json.loads(content)
                for k, v in data.items():
                    img_rel_path = v.get('id')  # 原始 id 是图片相对路径
                    v['id'] = k                  # 用 key 作为任务 id
                    if 'image_path' not in v:
                        if img_rel_path and args.origin_img_root:
                            v['image_path'] = os.path.join(args.origin_img_root, img_rel_path)
                        elif img_rel_path:
                            v['image_path'] = img_rel_path
                    tasks.append(v)
            else:
                tasks = [json.loads(line) for line in content.split('\n') if line.strip()]
    else:
        raise ValueError("Must specify --image_path + --prompt, or --metadata_file")

    # --- 分片 ---
    if world_size > 1:
        per_gpu = (len(tasks) + world_size - 1) // world_size
        local_tasks = tasks[rank * per_gpu: min((rank + 1) * per_gpu, len(tasks))]
    else:
        local_tasks = tasks

    if rank == 0:
        print(f"Total tasks: {len(tasks)}, this GPU: {len(local_tasks)}")

    # --- 推理 ---
    for i, task in enumerate(local_tasks):
        task_id = task.get('id', f'{i:04d}')
        image_path = task['image_path']
        prompt = task['prompt']
        outpath = os.path.join(args.output_dir, f"{task_id}.png")

        if os.path.exists(outpath):
            print(f"[GPU {rank}] Skipping {task_id}")
            continue

        print(f"[GPU {rank}] [{i+1}/{len(local_tasks)}] {task_id}: {prompt[:60]}...")
        try:
            src_image = pil_img2rgb(Image.open(image_path))
            if args.match_ref_size:
                target_size = compute_target_size_from_ref(src_image, args.resolution)
            else:
                target_size = (args.resolution, args.resolution)
            result = editing_image(
                gen_model=gen_model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                vae_model=vae_model,
                ref_image=src_image,
                prompt=prompt,
                ref_image_size=args.ref_image_size,
                ref_max_size=args.ref_max_size,
                ref_min_size=args.ref_min_size,
                ref_use_mar=args.ref_use_mar,
                ref_mar_resolution=args.ref_mar_resolution,
                target_size=target_size,
                num_timesteps=args.inference_steps,
                cfg_text_scale=args.cfg_text_scale,
                cfg_interval=[0, 1.0],
                cfg_renorm_min=args.cfg_renorm_min,
                timestep_shift=args.timestep_shift,
                device=device,
                device_type=device_type,
                seed=args.seed,
            )
            result.save(outpath)
            src_image.save(os.path.join(args.output_dir, f"{task_id}_src.png"))
            with open(os.path.join(args.output_dir, f"{task_id}_prompt.txt"), "w") as f:
                f.write(prompt)
            print(f"  Saved: {outpath}")
        except Exception as e:
            print(f"  Error on {task_id}: {e}")
            import traceback
            traceback.print_exc()

        device_info['empty_cache']()

    if rank == 0:
        print(f"Done! {len(local_tasks)} tasks processed.")
    if args.distributed:
        import torch.distributed as dist
        dist.barrier()


if __name__ == "__main__":
    main()
