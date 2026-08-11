# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import argparse

def extract_scores_and_average(entry: str) -> float:
    import re
    # 先用 **Label:** 5 格式（ours 用 gpt-4o-2024-11-20 返回的 markdown 格式）
    nums = re.findall(r'\*\*[^*]+\*\*[:\s]+(\d+(?:\.\d+)?)', entry)
    nums = [float(n) for n in nums if float(n) <= 10]
    if not nums:
        # 再用 Label: 5 / Label: 5 - explanation 格式（纯文本格式）
        for line in entry.splitlines():
            parts = line.strip().split(': ')
            if len(parts) >= 2:
                # 取冒号后第一个 token，可能是 "5" 或 "5 - explanation..."
                first = parts[1].strip().split()[0].rstrip('.,')
                if first.replace('.', '', 1).isdigit():
                    v = float(first)
                    if 1 <= v <= 10:
                        nums.append(v)
    if nums:
        return round(sum(nums) / len(nums), 2)
    return None

def compute_averages(result_json_dict):
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for each key and save it as a new JSON file")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result_json json')
    parser.add_argument('--average_score_json', type=str, required=True, help='Path of average_score_json json')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    averaged_data = compute_averages(data)

    missing = sorted(set(data) - set(averaged_data))
    if missing:
        raise RuntimeError(
            f"Could not parse judge scores for {len(missing)}/{len(data)} items; "
            f"keys include {missing[:20]}"
        )

    with open(args.average_score_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_data, f, indent=2)


if __name__ == '__main__':
    main()
