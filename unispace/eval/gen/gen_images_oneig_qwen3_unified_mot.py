"""
OneIG-Bench 图像生成脚本（Qwen3Unified RAE 架构）

照 gen_images_qwen3_unified_mot.py 的模式写，每个 prompt 生成 M×N 张图拼成网格。

输出目录结构（与 OneIG-Bench 评测脚本一致）：
    {output_dir}/{class_folder}/{model_name}/{id}.webp

    class_folder 映射：
        Anime_Stylization   → anime
        Portrait            → human
        General_Object      → object
        Text_Rendering      → text
        Knowledge_Reasoning → reasoning
        Multilingualism     → multilingualism  (ZH 模式)

用法（torchrun 多卡）：
    torchrun --nproc_per_node=8 eval/gen/gen_images_oneig_qwen3_unified_mot.py \\
        --mode EN \\
        --output_dir /path/to/output \\
        --model-path /path/to/ckpt \\
        --llm-path /path/to/Qwen3-8B \\
        --vae-path /path/to/rae_config.yaml \\
        --rae-accelerate-root /path/to/RAE_accelerate \\
        --distributed
"""

import os
import argparse
import gc
import random

import numpy as np
import torch
from PIL import Image
import pandas as pd

# ── 复用 gen_images_qwen3_unified_mot 的核心函数 ─────────────────────────────
from eval.gen.gen_images_qwen3_unified_mot import (
    get_device_info,
    load_model,
    generate_image,
)

import torch._dynamo
torch._dynamo.config.disable = True


# ── 常量 ─────────────────────────────────────────────────────────────────────

BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "OneIG-Benchmark",
)

CSV_EN = os.path.join(BENCHMARK_DIR, "OneIG-Bench.csv")
CSV_ZH = os.path.join(BENCHMARK_DIR, "OneIG-Bench-ZH.csv")

CATEGORY_TO_FOLDER = {
    "Anime_Stylization":   "anime",
    "Portrait":            "human",
    "General_Object":      "object",
    "Text_Rendering":      "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism":     "multilingualism",
}


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def make_grid(images: list, nrows: int, ncols: int) -> Image.Image:
    """将 nrows*ncols 张图片拼成网格，宽高取第一张的尺寸。"""
    assert len(images) == nrows * ncols, f"需要 {nrows*ncols} 张图，但有 {len(images)} 张"
    w, h = images[0].size
    grid = Image.new("RGB", (ncols * w, nrows * h))
    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        grid.paste(img.resize((w, h)), (col * w, row * h))
    return grid


def load_items(mode: str) -> pd.DataFrame:
    """加载 CSV，返回 DataFrame，统一列名为 prompt。"""
    if mode == "EN":
        df = pd.read_csv(CSV_EN)
        df["prompt"] = df["prompt_en"]
    elif mode == "ZH":
        df = pd.read_csv(CSV_ZH)
        df["prompt"] = df["prompt_cn"]
    else:
        raise ValueError(f"mode 必须是 EN 或 ZH，当前: {mode}")
    df["id"] = df["id"].astype(str).str.zfill(3)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OneIG-Bench generation with Qwen3Unified MOT")

    # 数据
    parser.add_argument("--mode", type=str, default="EN", choices=["EN", "ZH"],
                        help="评测语言（EN 或 ZH）")
    parser.add_argument("--benchmark-dir", type=str, default=BENCHMARK_DIR,
                        help="OneIG-Benchmark 根目录")

    # 输出
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True,
                        help="输出根目录，内部按 {class}/{model_name}/{id}.webp 组织")
    parser.add_argument("--model-name", type=str, default="unimm",
                        help="模型名（子目录名，对应 OneIG-Bench 评测脚本的 model_names 参数）")

    # 生成网格
    parser.add_argument("--grid-rows", type=int, default=2, help="网格行数（默认 2）")
    parser.add_argument("--grid-cols", type=int, default=2, help="网格列数（默认 2）")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="基础随机种子，第 i 张图用 base_seed + i")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每次 generate_image 调用生成的图片数（默认 4，即一次吃满整个 grid）")

    # 模型参数（同 gen_images_qwen3_unified_mot.py）
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--llm_type", type=str, default="qwen3")
    parser.add_argument("--vae-type", type=str, default="qwen3_unified")
    parser.add_argument("--rae-accelerate-root", type=str,
                        default=None)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--share_unified2llm", type=lambda x: x.lower() != 'false', default=False)
    parser.add_argument("--use_spatial_merge_und", type=lambda x: x.lower() != 'false', default=True)

    # 推理参数
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=10.0)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--timestep-shift", "--timestep_shift", type=float, default=0.112,
                        dest="timestep_shift")
    parser.add_argument("--inference_steps", type=int, default=50)

    # 多卡
    parser.add_argument("--distributed", action="store_true")

    args, _ = parser.parse_known_args()  # 忽略其他 bench 透传过来的未知参数（如 --match-ref-size）

    # ── 设备初始化 ─────────────────────────────────────────────────────────
    device_info = get_device_info()
    device_type = device_info['type']

    if args.distributed:
        import torch.distributed as dist
        dist.init_process_group(backend=device_info['backend'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device_info['set_device'](local_rank)
    device = f"{device_type}:{local_rank}"

    # ── 加载数据 ─────────────────────────────────────────────────────────
    df = load_items(args.mode)
    all_items = list(df.iterrows())

    # 按 rank 分片
    local_items = all_items[rank::world_size]
    if rank == 0:
        print(f"[rank {rank}] mode={args.mode}, total={len(all_items)}, local={len(local_items)}")
        print(f"[rank {rank}] grid={args.grid_rows}×{args.grid_cols} "
              f"({args.grid_rows * args.grid_cols} images per prompt)")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)

    num_per_prompt = args.grid_rows * args.grid_cols

    # ── 逐条生成 ──────────────────────────────────────────────────────────
    generated = skipped = errors = 0

    for _, row in local_items:
        category    = row["category"]
        prompt_id   = str(row["id"])
        prompt      = str(row["prompt"])
        class_folder = CATEGORY_TO_FOLDER.get(category, category.lower())

        out_dir  = os.path.join(args.output_dir, class_folder, args.model_name)
        out_path = os.path.join(out_dir, f"{prompt_id}.webp")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            images = []
            batch_size = min(args.batch_size, num_per_prompt)
            for batch_start in range(0, num_per_prompt, batch_size):
                cur_batch = min(batch_size, num_per_prompt - batch_start)
                seed = args.base_seed + int(prompt_id) * num_per_prompt + batch_start
                img_list = generate_image(
                    prompt=prompt,
                    num_timesteps=args.inference_steps,
                    cfg_scale=args.cfg_scale,
                    cfg_interval=args.cfg_interval,
                    cfg_renorm_min=args.cfg_renorm_min,
                    timestep_shift=args.timestep_shift,
                    resolution=args.resolution,
                    num_images=cur_batch,
                    device=device,
                    device_type=device_type,
                    gen_model=gen_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    vae_model=vae_model,
                    seed=seed,
                )
                images.extend(img_list)

            grid = make_grid(images, args.grid_rows, args.grid_cols)
            os.makedirs(out_dir, exist_ok=True)
            grid.save(out_path, format="webp", quality=95)

            generated += 1
            print(f"[rank {rank}] saved {class_folder}/{args.model_name}/{prompt_id}.webp"
                  f" | {prompt[:50]!r}")

        except Exception as e:
            errors += 1
            print(f"[rank {rank}] error on {prompt_id} ({category}): {e}")

        device_info['empty_cache']()
        gc.collect()

    print(f"[rank {rank}] done: generated={generated} skipped={skipped} errors={errors}")

    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
