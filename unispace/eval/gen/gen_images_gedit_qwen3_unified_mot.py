"""
GEdit-Bench 图像编辑生成脚本（Qwen3Unified RAE 架构）

基于 gen_images_imgedit_qwen3_unified_mot.py，适配 GEdit-Bench 数据集格式。

输出目录结构（与 run_gedit_score.py 期望一致）：
    {output_dir}/unimm/fullset/{task_type}/{instruction_language}/{key}.png

用法（torchrun 多卡）：
    torchrun --nproc_per_node=8 eval/gen/gen_images_gedit_qwen3_unified_mot.py \\
        --dataset-path /path/to/GEdit-Bench \\
        --output_dir /path/to/output \\
        --model-path /path/to/ckpt \\
        --llm-path /path/to/Qwen3-8B \\
        --vae-path /path/to/rae_config.yaml \\
        --rae-accelerate-root /path/to/RAE_accelerate \\
        --distributed
"""

import os
import argparse
import time
import random
import gc

import numpy as np
import torch
from PIL import Image
from io import BytesIO

# 复用 imgedit gen 脚本的所有公共函数
from eval.gen.gen_images_imgedit_qwen3_unified_mot import (
    get_device_info,
    load_model,
    editing_image,
    compute_target_size_from_ref,
)
from data.data_utils import pil_img2rgb

import torch._dynamo
torch._dynamo.config.disable = True


def main():
    parser = argparse.ArgumentParser(description="GEdit-Bench generation with Qwen3Unified MOT")
    # 数据
    parser.add_argument("--dataset-path", type=str,
                        required=True,
                        help="GEdit-Bench 本地 arrow 数据集目录")
    parser.add_argument("--language", type=str, default="all", choices=["all", "en", "cn"],
                        help="只跑指定语言（默认 all）")
    parser.add_argument("--task-type", type=str, default="all",
                        help="只跑指定 task type（默认 all）")
    # 输出
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出根目录，内部按 unimm/fullset/{task}/{lang}/{key}.png 组织")
    parser.add_argument("--model-name", type=str, default="unimm",
                        help="模型名（对应 run_gedit_score.py 的 model_name 参数）")
    # 模型（同 imgedit gen 脚本）
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--llm_type", type=str, default="qwen3")
    parser.add_argument("--vae-type", type=str, default="qwen3_unified")
    parser.add_argument("--rae-accelerate-root", type=str,
                        default=None)
    parser.add_argument("--max_latent_size", type=int, default=96)
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != 'false', default=False)
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != 'false', default=True)
    # 推理参数
    parser.add_argument("--ref_image_size", type=int, default=448)
    parser.add_argument("--ref-max-size", type=int, default=1024,
                        help="ref 图最大边长（保持宽高比），默认 1024 与训练一致")
    parser.add_argument("--ref-min-size", type=int, default=256)
    parser.add_argument("--ref-use-mar", action="store_true")
    parser.add_argument("--ref-mar-resolution", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=1024,
                        help="目标图分辨率（正方形边长），默认 1024")
    parser.add_argument("--match-ref-size", action="store_true",
                        help="按 ref 图宽高比生成（推荐）")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--timestep-shift", type=float, default=0.112)
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

    # --- 加载 GEdit-Bench 数据集 ---
    if rank == 0:
        print(f"Loading GEdit-Bench from {args.dataset_path} ...")
    import datasets as hf_datasets
    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset_path)
    # 禁用 Image 自动 decode，改为返回原始 bytes
    # 避免 HuggingFace Arrow 内部 image.load() 在坏图上崩进程
    dataset = dataset.cast_column('input_image', hf_datasets.Image(decode=False))
    dataset = dataset.cast_column('input_image_raw', hf_datasets.Image(decode=False))

    def decode_image_bytes(item, field='input_image_raw'):
        """从 bytes 手动 decode PIL Image，失败时 fallback 另一个 field"""
        for f in [field, 'input_image', 'input_image_raw']:
            try:
                raw = item[f]
                if isinstance(raw, dict):
                    data = raw.get('bytes') or raw.get('path')
                    if isinstance(data, bytes):
                        return Image.open(BytesIO(data)).convert('RGB')
                    elif isinstance(data, str):
                        return Image.open(data).convert('RGB')
                elif isinstance(raw, bytes):
                    return Image.open(BytesIO(raw)).convert('RGB')
            except Exception:
                continue
        raise RuntimeError(f"无法解码 key={item.get('key','?')} 的任何图片字段")

    def keep(item):
        if args.language != "all" and item['instruction_language'] != args.language:
            return False
        if args.task_type != "all" and item['task_type'] != args.task_type:
            return False
        return True

    tasks = [dataset[i] for i in range(len(dataset)) if keep(dataset[i])]

    if rank == 0:
        print(f"Total tasks after filter: {len(tasks)}")

    # --- 分片给各 GPU ---
    if world_size > 1:
        per_gpu = (len(tasks) + world_size - 1) // world_size
        local_tasks = tasks[rank * per_gpu: min((rank + 1) * per_gpu, len(tasks))]
    else:
        local_tasks = tasks

    if rank == 0:
        print(f"This GPU: {len(local_tasks)} tasks")

    # --- 加载模型 ---
    if rank == 0:
        print("Loading model...")
    t0 = time.time()
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    if rank == 0:
        print(f"Model loaded in {time.time() - t0:.1f}s")

    # --- 生成 ---
    for i, item in enumerate(local_tasks):
        task_type = item['task_type']
        key = item['key']
        instruction = item['instruction']
        lang = item['instruction_language']

        out_dir = os.path.join(args.output_dir, args.model_name, 'fullset', task_type, lang)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{key}.png")

        if os.path.exists(out_path):
            if rank == 0 and i < 5:
                print(f"[GPU {rank}] Skip {task_type}/{lang}/{key}")
            continue

        print(f"[GPU {rank}] [{i+1}/{len(local_tasks)}] {task_type}/{lang}/{key[:16]}... | {instruction[:60]}")

        try:
            # 手动解码：decode=False 模式下 item 里存的是 bytes，用 decode_image_bytes 处理
            src_image = decode_image_bytes(item, field='input_image_raw')

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
                prompt=instruction,
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
            result.save(out_path)
            # 保存原图和 prompt（同 imgedit gen 脚本）
            src_image.save(os.path.join(out_dir, f"{key}_src.png"))
            with open(os.path.join(out_dir, f"{key}_prompt.txt"), "w") as fp:
                fp.write(instruction)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Error on {key}: {e}")
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
