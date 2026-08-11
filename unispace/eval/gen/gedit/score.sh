#!/bin/bash
# 用法: bash score.sh <edited_images_dir> <save_dir> [backbone] [gpt_model] [model_name]
# 例如（gpt4o）:  bash eval/gen/gedit/score.sh images/ scores/
# 例如（gpt-4.1）: bash eval/gen/gedit/score.sh images/ scores/ gpt41 gpt-4.1
# 例如（sensenova）: bash eval/gen/gedit/score.sh images/ scores/ gpt4o gpt-4o-2024-05-13 sensenova_u1

set -e

EDITED_IMAGES_DIR=$1
SAVE_DIR=$2
BACKBONE=${3:-gpt4o}
GPT_MODEL=${4:-gpt-4o-2024-05-13}
MODEL_NAME=${5:-unimm}
MAX_WORKERS=${6:-6}
DATASET_PATH=${GEDIT_DATASET_PATH:?Set GEDIT_DATASET_PATH to the local GEdit-Bench directory}
PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

cd "$PROJECT_ROOT"

: "${WISE_API_KEY:?Set WISE_API_KEY for the evaluator}"
: "${WISE_API_BASE:?Set WISE_API_BASE for the evaluator}"

echo "=== GEdit Scoring (backbone=$BACKBONE model=$GPT_MODEL model_name=$MODEL_NAME) ==="
$PYTHON "$SCRIPT_DIR/run_gedit_score.py" \
    --model_name "$MODEL_NAME" \
    --edited_images_dir "$EDITED_IMAGES_DIR" \
    --save_dir "$SAVE_DIR" \
    --dataset-path "$DATASET_PATH" \
    --backbone "$BACKBONE" \
    --gpt-model "$GPT_MODEL" \
    --max-workers "$MAX_WORKERS"

echo "=== GEdit Statistics ==="
RESULT_FILE="$SAVE_DIR/gedit_results_${BACKBONE}.txt"
$PYTHON "$SCRIPT_DIR/calculate_statistics.py" \
    --model_name "$MODEL_NAME" \
    --save_path "$SAVE_DIR" \
    --backbone "$BACKBONE" \
    --language all \
    --json-output "$SAVE_DIR/metrics.json" \
    | tee "$RESULT_FILE"
echo "Results saved to $RESULT_FILE"
