# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import time
import random

import numpy as np
import torch
from PIL import Image

import os as _os
import importlib as _importlib
_gen_module_name = _os.environ.get("UNIMM_GEN_MODULE", "eval.gen.gen_images")
_gen_mod = _importlib.import_module(_gen_module_name)
get_device_info = _gen_mod.get_device_info
load_model = _gen_mod.load_model
generate_image = _gen_mod.generate_image


def stitch_images_2x2(images):
    if len(images) != 4:
        raise ValueError("Need exactly 4 images for 2x2 grid")
    images = [img.crop(img.getbbox()) for img in images]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    max_height = max(heights)
    grid_image = Image.new('RGB', (max_width * 2, max_height * 2))
    positions = [(0, 0), (max_width, 0), (0, max_height), (max_width, max_height)]
    for img, pos in zip(images, positions):
        grid_image.paste(img, pos)
    return grid_image


def load_prompts_from_folder(prompts_folder):
    prompt_files = sorted([f for f in os.listdir(prompts_folder) if f.endswith('.txt')])
    prompts_data = []
    for txt_file in prompt_files:
        file_path = os.path.join(prompts_folder, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        file_id = os.path.splitext(txt_file)[0]
        prompts_data.append({'id': file_id, 'prompt': prompt})
    return prompts_data


def main():
    parser = argparse.ArgumentParser(description="Generate images for DPG benchmark")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompts_folder", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--llm-path', type=str, required=True)
    parser.add_argument('--vae-path', type=str, default=None)
    parser.add_argument('--vae-type', type=str, default='flux1-vae', choices=['flux1-vae', 'flux2-vae', 'qwen3_unified'])
    parser.add_argument('--inference-mode', type=str, default='flash')
    parser.add_argument('--inference-steps', type=int, default=50)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--use_spatial_merge", action="store_true", default=False)
    parser.add_argument("--use_spatial_merge_gen", type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--rae-accelerate-root", type=str, default=None)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--pred_type", type=str, default="v_pred")
    parser.add_argument("--llm_type", type=str, default="qwen3")
    args, _ = parser.parse_known_args()

    if args.num_images != 4:
        print(f"Warning: num_images changed from {args.num_images} to 4 for 2x2 grid")
        args.num_images = 4

    device_info = get_device_info()
    device_type = device_info['type']

    seed = args.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        print("Loading model...")
    start_time = time.time()
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    if rank == 0:
        print(f"Model loaded in {time.time() - start_time:.2f}s")

    prompts_data = load_prompts_from_folder(args.prompts_folder)
    total_prompts = len(prompts_data)

    if world_size > 1:
        per_gpu = (total_prompts + world_size - 1) // world_size
        start_idx = rank * per_gpu
        end_idx = min(start_idx + per_gpu, total_prompts)
        local_items = list(range(start_idx, end_idx))
    else:
        local_items = list(range(total_prompts))

    if rank == 0:
        print(f"Generating {total_prompts} prompts, {args.num_images} images each (2x2 grid)")

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = args.timestep_shift
    num_timesteps = args.inference_steps
    cfg_renorm_min = 0.0
    batch_size = min(args.batch_size, args.num_images)

    total_start = time.time()

    for prompt_idx in local_items:
        prompt_data = prompts_data[prompt_idx]
        prompt_id = prompt_data['id']
        prompt = prompt_data['prompt']

        output_file = os.path.join(output_dir, f"{prompt_id}.jpg")
        if os.path.exists(output_file):
            print(f"[GPU {rank}] Skipping {prompt_id}")
            continue

        print(f"[GPU {rank}] [{prompt_idx}/{total_prompts}] {prompt[:60]}...")

        all_images = []
        for batch_start in range(0, args.num_images, batch_size):
            cur_batch = min(batch_size, args.num_images - batch_start)
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

        if len(all_images) == 4:
            grid_image = stitch_images_2x2(all_images)
            grid_image.save(output_file, format='JPEG', quality=95)
            print(f"  Saved grid to {output_file}")
        elif all_images:
            for i, img in enumerate(all_images):
                img.save(os.path.join(output_dir, f"{prompt_id}_{i}.jpg"), format='JPEG', quality=95)

        device_info['empty_cache']()

    total_time = time.time() - total_start
    if local_items:
        print(f"[GPU {rank}] Done! {len(local_items)} prompts in {total_time:.1f}s")

    if args.distributed and rank == 0:
        print("All GPUs completed.")


if __name__ == "__main__":
    main()
