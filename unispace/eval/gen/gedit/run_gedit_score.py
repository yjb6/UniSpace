"""
GEdit-Bench VIEScore 打分脚本（适配版）

改动：
  - load_from_disk 替代 load_dataset（使用本地 arrow）
  - 去掉 megfile，用 os/glob 代替
  - API endpoint/key 复用 WISE 环境变量（WISE_API_KEY / WISE_API_BASE）
  - GPT 模型名改为 gpt-4o-2024-05-13（与 WISE proxy 兼容）

用法：
    WISE_API_KEY=xxx WISE_API_BASE=https://... \\
    python eval/gen/gedit/run_gedit_score.py \\
        --model_name unimm \\
        --edited_images_dir /path/to/output_dir \\
        --save_dir /path/to/score_dir \\
        --dataset-path /path/to/GEdit-Bench \\
        --backbone gpt4o
"""

import sys
import os
import glob
import csv
import json
import time
import argparse
import math
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

# ── 本地 viescore（eval/gen/gedit/viescore/）优先 ────────────────────────
_LOCAL_GEDIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LOCAL_GEDIT)
# viescore/__init__.py 里有 `from utils import ...` 的老式绝对导入，
# 需要把 viescore/ 目录本身也加进 path
sys.path.insert(0, os.path.join(_LOCAL_GEDIT, 'viescore'))

_WISE_BASE = os.environ.get('WISE_API_BASE')
_WISE_KEY = os.environ.get('WISE_API_KEY')

from openai import OpenAI
from viescore import VIEScore


def _make_vie_score(backbone, gpt_model="gpt-4o-2024-05-13"):
    """实例化 VIEScore，用 OpenAI SDK 替换 get_parsed_output（同 gpt_eval_wise.py）"""
    if not _WISE_BASE or not _WISE_KEY:
        raise RuntimeError("Set WISE_API_BASE and WISE_API_KEY before running GEdit scoring.")
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False)
    tmp.write(_WISE_KEY + '\n'); tmp.close()
    # 固定传 gpt4o 让 VIEScore 初始化不报错（实际调用已被 OpenAI SDK 替换）
    vs = VIEScore(backbone="gpt4o", task="tie", key_path=tmp.name)
    os.unlink(tmp.name)

    _client = OpenAI(api_key=_WISE_KEY, base_url=_WISE_BASE)

    def _get_parsed_output(prompt):
        try:
            resp = _client.chat.completions.create(
                model=gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1400,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Failed to obtain answer via API: {e}"

    vs.model.get_parsed_output = _get_parsed_output
    return vs


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_src_image(item):
    """从 dataset item 加载原图，优先 input_image_raw，失败 fallback input_image。
    dataset 以 decode=False 加载，字段为 {'bytes': ..., 'path': ...} 或直接 bytes。"""
    from io import BytesIO
    for field in ['input_image_raw', 'input_image']:
        try:
            raw = item.get(field)
            if raw is None:
                continue
            if isinstance(raw, dict):
                data = raw.get('bytes') or raw.get('path')
            else:
                data = raw
            if isinstance(data, bytes):
                img = Image.open(BytesIO(data)).convert('RGB')
            elif isinstance(data, str):
                img = Image.open(data).convert('RGB')
            else:
                continue
            return img
        except Exception:
            continue
    raise RuntimeError(f"无法解码 key={item.get('key','?')} 的任何图片字段")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return int(width), int(height), int(width * height)


def find_files_with_given_basename(folder_path, basename):
    """在 folder_path 下找匹配 basename.* 的文件，返回文件名列表（去路径）"""
    pattern = os.path.join(folder_path, f"{basename}.*")
    return [os.path.basename(f) for f in glob.glob(pattern)]


def _load_checkpoint(jsonl_path):
    """从 jsonl checkpoint 加载已完成结果，返回 {key: result} dict。
    文件不存在或行解析失败时静默跳过，保证 resume 鲁棒性。"""
    if not os.path.exists(jsonl_path):
        return {}
    results = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                results[r['key']] = r
            except Exception:
                pass
    return results


def process_single_item(item, vie_score, max_retries=12):
    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    edit_image_path = item['edited_image_path']

    for retry in range(max_retries):
        try:
            pil_image = load_src_image(item)   # raw 优先，失败 fallback 小图
            pil_image_edited = Image.open(edit_image_path).convert("RGB")
            src_w, src_h, _ = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
            edt_w, edt_h, _ = calculate_dimensions(512 * 512, pil_image_edited.width / pil_image_edited.height)
            pil_image = pil_image.resize((src_w, src_h))
            pil_image_edited = pil_image_edited.resize((edt_w, edt_h))
            score_list = vie_score.evaluate([pil_image, pil_image_edited], instruction)
            sem, qual, overall = score_list
            print(f"sem={sem:.2f} qual={qual:.2f} overall={overall:.2f} | {instruction_language} | {instruction[:60]}")
            return {
                "key": key,
                "edited_image": edit_image_path,
                "instruction": instruction,
                "sementics_score": sem,
                "quality_score": qual,
                "intersection_exist": item['Intersection_exist'],
                "instruction_language": instruction_language,
            }
        except Exception as e:
            if retry < max_retries - 1:
                wait = min((retry + 1) * 2, 60)
                print(f"Error (attempt {retry+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed after {max_retries} attempts for key={key}: {e}")
                return None


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unimm")
    parser.add_argument("--edited_images_dir", type=str, required=True,
                        help="gen 脚本的 output_dir（内含 {model_name}/fullset/...）")
    parser.add_argument("--instruction_language", type=str, default="all",
                        choices=["all", "en", "cn"])
    parser.add_argument("--task_type", type=str, default="all")
    parser.add_argument("--save_dir", type=str, default="gedit_csv_results")
    parser.add_argument("--backbone", type=str, default="gpt4o",
                        help="存结果的子目录名，如 gpt4o / gpt41")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o-2024-05-13",
                        help="实际调用的 GPT model name，如 gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="并发线程数（默认 6）")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Local path created by datasets.save_to_disk for GEdit-Bench")
    args = parser.parse_args()

    ALL_GROUPS = ["background_change", "color_alter", "material_alter", "motion_change",
                  "ps_human", "style_change", "subject-add", "subject-remove",
                  "subject-replace", "text_change", "tone_transfer"]
    groups = ALL_GROUPS if args.task_type == "all" else [args.task_type]

    # 加载数据集（decode=False：避免坏图在 Arrow decode 时崩进程）
    import datasets as hf_datasets
    print(f"Loading GEdit-Bench from {args.dataset_path} ...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column('input_image', hf_datasets.Image(decode=False))
    dataset = dataset.cast_column('input_image_raw', hf_datasets.Image(decode=False))
    dataset_by_group = defaultdict(list)
    for item in tqdm(dataset, desc="Grouping dataset"):
        if args.instruction_language == "all" or item['instruction_language'] == args.instruction_language:
            dataset_by_group[item['task_type']].append(item)
    for k, v in dataset_by_group.items():
        print(f"  {k}: {len(v)} samples")

    # 初始化 VIEScore
    vie_score = _make_vie_score(args.backbone, gpt_model=args.gpt_model)

    save_path = os.path.join(args.save_dir, args.model_name, args.backbone)
    os.makedirs(save_path, exist_ok=True)

    # 所有 group 一次性提交，全并发（max_workers 控制并发数）
    # resume 粒度：group 级（CSV 已存在 → 跳过整 group）+ item 级（jsonl checkpoint）
    pending_groups = []
    group_checkpoints = {}   # group_name -> {key: result}
    for group_name in groups:
        group_csv_path = os.path.join(
            save_path,
            f"{args.model_name}_{group_name}_{args.instruction_language}_vie_score.csv"
        )
        if os.path.exists(group_csv_path):
            with open(group_csv_path, newline='') as _f:
                existing_rows = list(csv.DictReader(_f))
            expected_keys = {str(item['key']) for item in dataset_by_group[group_name]}
            existing_keys = [str(row.get('key', '')) for row in existing_rows]
            if (set(existing_keys) != expected_keys or
                    len(existing_keys) != len(set(existing_keys))):
                missing = sorted(expected_keys - set(existing_keys))
                extra = sorted(set(existing_keys) - expected_keys)
                raise RuntimeError(
                    f"Invalid existing CSV for {group_name}: rows={len(existing_keys)} "
                    f"expected={len(expected_keys)} missing={missing[:20]} "
                    f"extra={extra[:20]}"
                )
            print(f"  {group_name}: complete CSV exists ({len(existing_keys)} rows), skipping")
            continue
        # 加载 item 级 checkpoint（jsonl），支持中途续跑
        jsonl_path = os.path.join(
            save_path,
            f"{args.model_name}_{group_name}_{args.instruction_language}_checkpoint.jsonl"
        )
        existing = _load_checkpoint(jsonl_path)
        if existing:
            print(f"  {group_name}: {len(existing)} items already scored (checkpoint), resuming...")
        group_checkpoints[group_name] = existing
        pending_groups.append(group_name)

    # 收集所有待处理 item，跳过 checkpoint 中已完成的 key
    all_tasks = []
    for group_name in pending_groups:
        done_keys = set(group_checkpoints[group_name].keys())
        for item in dataset_by_group[group_name]:
            key = item['key']
            if key in done_keys:
                continue
            lang = item['instruction_language']
            img_dir = os.path.join(
                args.edited_images_dir, args.model_name, 'fullset', group_name, lang)
            candidates = find_files_with_given_basename(img_dir, key)
            if not candidates:
                print(f"  Warning: {key} not found in {img_dir}")
                continue
            item = dict(item)
            item['edited_image_path'] = os.path.join(img_dir, candidates[0])
            all_tasks.append((group_name, item))

    total_checkpoint = sum(len(v) for v in group_checkpoints.values())
    print(f"总待评测: {len(all_tasks)} 条（已完成 {total_checkpoint} 条），并发数: {args.max_workers}")

    # group_results 预填充 checkpoint 已有结果
    group_results = {g: list(group_checkpoints[g].values()) for g in pending_groups}

    # 每个 group 一把锁，用于 jsonl 追加写入（多线程安全）
    group_jsonl_locks = {g: threading.Lock() for g in pending_groups}
    group_jsonl_paths = {
        g: os.path.join(
            save_path,
            f"{args.model_name}_{g}_{args.instruction_language}_checkpoint.jsonl"
        )
        for g in pending_groups
    }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_group = {
            executor.submit(process_single_item, item, vie_score): group_name
            for group_name, item in all_tasks
        }
        for future in tqdm(as_completed(future_to_group), total=len(future_to_group),
                           desc="Scoring all groups"):
            group_name = future_to_group[future]
            result = future.result()
            if result:
                group_results[group_name].append(result)
                # 立即追加到 jsonl checkpoint（中断后可 resume）
                with group_jsonl_locks[group_name]:
                    with open(group_jsonl_paths[group_name], 'a') as _f:
                        _f.write(json.dumps(result, ensure_ascii=False) + '\n')

    for group_name in pending_groups:
        group_csv_path = os.path.join(
            save_path,
            f"{args.model_name}_{group_name}_{args.instruction_language}_vie_score.csv"
        )
        # 保存 CSV
        rows = group_results[group_name]
        expected_keys = {str(item['key']) for item in dataset_by_group[group_name]}
        result_keys = [str(row.get('key', '')) for row in rows]
        if set(result_keys) != expected_keys or len(result_keys) != len(set(result_keys)):
            missing = sorted(expected_keys - set(result_keys))
            extra = sorted(set(result_keys) - expected_keys)
            raise RuntimeError(
                f"Judge coverage is incomplete for {group_name}: "
                f"rows={len(result_keys)} expected={len(expected_keys)} "
                f"missing={missing[:20]} extra={extra[:20]}"
            )
        with open(group_csv_path, 'w', newline='') as f:
            fieldnames = ["key", "edited_image", "instruction", "sementics_score",
                          "quality_score", "intersection_exist", "instruction_language"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  Saved {group_name}: {len(rows)} samples → {group_csv_path}")
