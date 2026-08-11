# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

lock = threading.Lock()  # For thread-safe file writing


def has_scores(text):
    """Reject API refusals and malformed judge responses before checkpointing."""
    markdown = re.findall(r'\*\*[^*]+\*\*[:\s]+(\d+(?:\.\d+)?)', text)
    plain = re.findall(r'^\s*[^:\n]+:\s*(\d+(?:\.\d+)?)', text, flags=re.M)
    values = [float(value) for value in (markdown or plain)]
    return len([value for value in values if 1 <= value <= 10]) >= 3

def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type, prompts):
    import time as _time
    original_image_base64 = image_to_base64(original_image_path)
    result_image_base64 = image_to_base64(result_image_path)

    if not original_image_base64 or not result_image_base64:
        return {"error": "Image conversion failed"}

    prompt = prompts[edit_type]
    full_prompt = prompt.replace('<edit_prompt>', edit_prompt)
    full_prompt += (
        "\nImportant: Always return every numeric score requested by the rubric, "
        "even when the edited image completely violates the task premise (for "
        "example, an extraction has a non-white background). In that case, assign "
        "the lowest scores justified by the rubric; do not refuse or omit fields."
    )

    attempt = 0
    while attempt < max_retries:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                    ]
                }]
            )
            content = response.choices[0].message.content
            if not has_scores(content):
                raise RuntimeError(f"Judge returned no valid score triplet: {content[:160]!r}")
            return response
        except Exception as e:
            wait = min(2 ** attempt + 5, 60)
            print(f"Retry {attempt+1} after {wait}s: {e}")
            _time.sleep(wait)
            attempt += 1
    raise RuntimeError(f"Judge request failed after {max_retries} attempts")

def save_result_jsonl(result, key, output_jsonl_path):
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {
                "key": key,
                "result": result
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys(jsonl_path):
    processed_keys = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if has_scores(data.get("result", "")):
                        processed_keys.add(data["key"])
                except Exception as e:
                    print(f"Error loading line: {e}")
    return processed_keys

def collect_jsonl_to_dict(jsonl_path):
    result_dict = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result = data.get("result", "")
                    # A retry log may contain an earlier refusal followed by a
                    # valid answer for the same key.  Never let malformed
                    # historical lines make the final JSON look complete.
                    if has_scores(result):
                        result_dict[data["key"]] = result
                except Exception as e:
                    print(f"Error parsing line: {e}")
    return result_dict

def process_single_item(key, item, result_img_folder, origin_img_root, prompts, output_jsonl_path):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gpt(origin_img_path, result_img_path, edit_prompt, edit_type, prompts)
    # Ensure 'choices' attribute exists in response
    result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    save_result_jsonl(result, key, output_jsonl_path)
    return key, result

def process_json(edit_json, result_img_folder, origin_img_root, num_threads, prompts):
    output_jsonl_path = os.path.join(result_img_folder, 'result.jsonl')
    output_json_path = os.path.join(result_img_folder, 'result.json')
    with open(edit_json, 'r') as f:
        edit_infos = json.load(f)
    # Load already processed keys
    processed_keys = load_processed_keys(output_jsonl_path)
    print(f"{len(processed_keys)} items already processed, {len(edit_infos) - len(processed_keys)} remaining...")
    # Filter out tasks that have already been processed
    left_edit_infos = {k: v for k, v in edit_infos.items() if k not in processed_keys}
    total = len(left_edit_infos)
    if total == 0:
        print("Nothing to process. All items are completed.")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_key = {
                executor.submit(process_single_item, key, item, result_img_folder, origin_img_root, prompts, output_jsonl_path): key
                for key, item in left_edit_infos.items()
            }
            for future in tqdm(as_completed(future_to_key), total=total, desc="Processing edits"):
                key = future_to_key[future]
                try:
                    future.result()  # Already saved in jsonl
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    # Failed keys will not be saved to jsonl
    # After all finished, collect jsonl to dict and save to json
    final_results = collect_jsonl_to_dict(output_jsonl_path)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"All processing completed. Final result saved in {output_json_path}")
    missing = sorted(set(edit_infos) - set(final_results))
    if missing:
        raise RuntimeError(
            f"Judge coverage is incomplete: {len(final_results)}/{len(edit_infos)}; "
            f"missing keys include {missing[:20]}"
        )

def main():
    global openai_client, model, max_retries
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts")
    parser.add_argument('--api_key', type=str, default=os.environ.get("WISE_API_KEY"), required=os.environ.get("WISE_API_KEY") is None)
    parser.add_argument('--api_base', type=str, default=os.environ.get("WISE_API_BASE"), required=os.environ.get("WISE_API_BASE") is None)
    parser.add_argument('--model', type=str, default="gpt-4o-2024-11-20")
    parser.add_argument('--max_retries', type=int, default=12)
    args = parser.parse_args()

    model = args.model
    max_retries = args.max_retries
    openai_client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=120.0, max_retries=0)

    prompts = load_prompts(args.prompts_json)
    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes, prompts)

if __name__ == "__main__":
    main()
