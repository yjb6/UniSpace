# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
GenEval 专用图像生成脚本 — 直接输出 evaluate_images_mp.py 期望的目录格式。

输出格式:
    {output_dir}/
    ├── 0/
    │   ├── metadata.jsonl      ← 原始 GenEval metadata (含 tag, include, prompt)
    │   └── samples/
    │       ├── 0.png
    │       ├── 1.png
    │       └── ...
    ├── 1/
    │   ├── metadata.jsonl
    │   └── samples/
    │       └── ...

用法:
    torchrun --nproc_per_node=1 -m eval.gen.gen_images_geneval \
        --output_dir ./output/geneval \
        --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \
        --num_images_per_prompt 4 --batch_size 4 \
        --model-path /path/to/ckpt --llm-path /path/to/qwen3vl --vae-path /path/to/ae.safetensors
"""

import argparse
import json
import os
import time
import random

import numpy as np
import torch

import os as _os
import importlib as _importlib
_gen_module_name = _os.environ.get("UNIMM_GEN_MODULE", "eval.gen.gen_images")
_gen_mod = _importlib.import_module(_gen_module_name)
get_device_info = _gen_mod.get_device_info
load_model = _gen_mod.load_model
generate_image = _gen_mod.generate_image


def main():
    parser = argparse.ArgumentParser(description="GenEval image generation (native format output)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="GenEval evaluation_metadata.jsonl")
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--vae-type", type=str, default="flux1-vae", choices=["flux1-vae", "flux2-vae", "qwen3_unified"])
    parser.add_argument("--inference-mode", type=str, default="flash")
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--mrope_section", type=str, default="")
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--use_spatial_merge", action="store_true", default=False)
    parser.add_argument("--use_spatial_merge_gen", type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--rae-accelerate-root", type=str, default=None)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--pred_type", type=str, default="v_pred")
    parser.add_argument("--llm_type", type=str, default="qwen3")
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------
    # 设备检测
    # ------------------------------------------------------------------
    device_info = get_device_info()
    device_type = device_info['type']

    # ------------------------------------------------------------------
    # 随机种子
    # ------------------------------------------------------------------
    seed = args.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        elif device_type == 'npu':
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 分布式
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 输出目录
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    if rank == 0:
        print("Loading model...")
    start_time = time.time()
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    if rank == 0:
        print(f"Model loaded in {time.time() - start_time:.2f}s")

    # ------------------------------------------------------------------
    # 加载 GenEval metadata
    # ------------------------------------------------------------------
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp if line.strip()]
    prompts = [m.get("prompt", "") for m in metadatas]
    if rank == 0:
        print(f"Loaded {len(prompts)} prompts from {args.metadata_file}")

    # ------------------------------------------------------------------
    # 分配到各 GPU
    # ------------------------------------------------------------------
    if world_size > 1:
        per_gpu = (len(prompts) + world_size - 1) // world_size
        start_idx = rank * per_gpu
        end_idx = min(start_idx + per_gpu, len(prompts))
        local_items = list(range(start_idx, end_idx))
    else:
        local_items = list(range(len(prompts)))

    if rank == 0:
        print(f"Generating {len(prompts)} prompts, {args.num_images_per_prompt} images each")

    # ------------------------------------------------------------------
    # 生成参数
    # ------------------------------------------------------------------
    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = args.timestep_shift
    num_timesteps = args.inference_steps
    cfg_renorm_min = 0.9
    num_images_per_prompt = args.num_images_per_prompt
    batch_size = min(args.batch_size, num_images_per_prompt)

    total_start = time.time()

    # ------------------------------------------------------------------
    # 生成循环
    # ------------------------------------------------------------------
    for prompt_idx in local_items:
        prompt = prompts[prompt_idx]
        metadata = metadatas[prompt_idx]

        print(f"[GPU {rank}] [{prompt_idx}/{len(prompts)}] {prompt[:60]}...")

        # GenEval 格式: {output_dir}/{prompt_idx}/samples/
        prompt_dir = os.path.join(output_dir, str(prompt_idx))
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # 保存 metadata.jsonl
        metadata_path = os.path.join(prompt_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)

        # 断点续跑: 检查已生成的图片数
        existing = [f for f in os.listdir(samples_dir) if f.endswith('.png')]
        if len(existing) >= num_images_per_prompt:
            print(f"  Skipping (already {len(existing)} images)")
            continue

        # 分批生成
        all_images = []
        for batch_start in range(0, num_images_per_prompt, batch_size):
            cur_batch = min(batch_size, num_images_per_prompt - batch_start)
            t0 = time.time()
            try:
                images = generate_image(
                    prompt=prompt,
                    cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    num_images=cur_batch,
                    resolution=args.resolution,
                    inference_mode=args.inference_mode,
                    device=device,
                    device_type=device_type,
                    gen_model=gen_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    vae_model=vae_model,
                    seed=seed,
                )
                all_images.extend(images)
                print(f"  Batch {batch_start//batch_size+1} done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                break

        # 保存图片: samples/0.png, samples/1.png, ...
        for img_idx, image in enumerate(all_images):
            try:
                bbox = image.getbbox()
                if bbox:
                    image = image.crop(bbox)
            except Exception:
                pass
            image.save(os.path.join(samples_dir, f"{img_idx}.png"))

        print(f"  Saved {len(all_images)} images to {samples_dir}")
        device_info['empty_cache']()

    total_time = time.time() - total_start
    if len(local_items) > 0:
        print(f"[GPU {rank}] Done! {len(local_items)} prompts in {total_time:.1f}s")

    if args.distributed and rank == 0:
        print("All GPUs completed.")


if __name__ == "__main__":
    main()
