#!/usr/bin/env python3
"""Create a strict machine-readable OneIG summary from the five score CSVs."""
import argparse, csv, json
from pathlib import Path

COLUMNS = {"alignment": "alignment", "text": "text score", "reasoning": "reasoning",
           "style": "style", "diversity": "total average"}

def latest_score(result_dir, metric, mode, model_name):
    files = sorted(result_dir.glob(f"{metric}_score_{mode}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"missing {metric} score CSV for {mode}")
    with files[-1].open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    index_column = next((key for key in rows[0] if not key), None) if rows else None
    for row in rows:
        row_model = row.get(index_column, "") if index_column is not None else ""
        if row_model == model_name or (len(rows) == 1 and not row_model):
            return float(row[COLUMNS[metric]])
    raise ValueError(f"model {model_name!r} missing from {files[-1]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["EN", "ZH"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = {metric: latest_score(args.result_dir, metric, args.mode, args.model_name) for metric in COLUMNS}
    scores["overall"] = sum(scores.values()) / len(COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
