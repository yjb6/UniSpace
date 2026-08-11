#!/bin/bash
# OneIG-Bench 打分脚本
# 用法: bash eval/gen/oneig/score.sh <gen_output_dir> <result_dir> [mode=EN] [model_name=unimm]
#
# 统一使用 rae 环境（需要 dreamsim, megfile 等）。
#
# 从项目根目录运行:
#   bash eval/gen/oneig/score.sh \
#       exp/rae-unified/eval/results/oneig_en/0060000/oneig/images \
#       exp/rae-unified/eval/results/oneig_en/0060000/oneig \
#       EN unimm

set -eo pipefail

GEN_OUTPUT_DIR=$(realpath "$1")
RESULT_DIR=$(realpath "$2")
MODE=${3:-EN}
MODEL_NAME=${4:-unimm}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BENCHMARK_DIR="${ONEIG_BENCH_ROOT:-${PROJECT_ROOT}/OneIG-Benchmark}"
PYTHON="${PYTHON:-python}"

# Qwen2.5-VL-7B（alignment / text 打分）
export ONEIG_VL_MODEL="${ONEIG_VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
# OneIG-StyleEncoder（style 打分）
export ONEIG_SE_MODEL="${ONEIG_SE_MODEL:-xingpng/OneIG-StyleEncoder}"
# LLM2CLIP（reasoning 打分，本地已有）
export ONEIG_CLIP_PROC="${ONEIG_CLIP_PROC:-openai/clip-vit-large-patch14-336}"
export ONEIG_LLM2CLIP_CLIP="${ONEIG_LLM2CLIP_CLIP:-microsoft/LLM2CLIP-Openai-L-14-336}"
# LLM2CLIP-Llama 是 HF cache 格式，指到 snapshots 子目录
export ONEIG_LLM2CLIP_LLM="${ONEIG_LLM2CLIP_LLM:-microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned}"
# dreamsim 模型目录（diversity 打分）
export TORCH_HOME="${BENCHMARK_DIR}/models"
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULT_DIR/logs"
RUN_MARKER="$RESULT_DIR/.oneig_score_started"
touch "$RUN_MARKER"
# DolphinFS may expose second-level mtimes. Keep subsequent score files
# strictly newer than the marker used by find -newer.
sleep 1

copy_fresh_scores() {
    local prefix=$1 found=0 file
    while IFS= read -r -d '' file; do
        cp "$file" "$RESULT_DIR/"
        found=1
    done < <(find "${BENCHMARK_DIR}/results" -maxdepth 1 -type f \
        -name "${prefix}*${MODE}*.csv" -newer "$RUN_MARKER" -print0)
    if [ "$found" -ne 1 ]; then
        echo "ERROR: no fresh ${prefix} ${MODE} score CSV was produced" >&2
        return 1
    fi
}

echo "=== OneIG-Bench Scoring ==="
echo "mode:         $MODE"
echo "model_name:   $MODEL_NAME"
echo "gen_output:   $GEN_OUTPUT_DIR"
echo "result_dir:   $RESULT_DIR"
echo ""

cd "$BENCHMARK_DIR"

# ── class_items 按 mode 区分 ──────────────────────────────────────────────────
if [ "$MODE" = "ZH" ]; then
    ALIGN_CLASSES="anime human object multilingualism"
    DIV_CLASSES="anime human object text reasoning multilingualism"
else
    ALIGN_CLASSES="anime human object"
    DIV_CLASSES="anime human object text reasoning"
fi

# ── 1. Alignment Score ────────────────────────────────────────────────────────
echo "=== [1/5] Alignment Score ==="
if ls "$RESULT_DIR"/alignment_score_${MODE}_*.csv 2>/dev/null | grep -q .; then
    echo "[skip] alignment already done"
else
    $PYTHON -m scripts.alignment.alignment_score \
        --mode "$MODE" \
        --image_dirname "$GEN_OUTPUT_DIR" \
        --model_names "$MODEL_NAME" \
        --image_grid "2,2" \
        --class_items $ALIGN_CLASSES \
        2>&1 | tee "$RESULT_DIR/logs/alignment.log"
    copy_fresh_scores alignment
fi

# ── 2. Text Score ─────────────────────────────────────────────────────────────
echo "=== [2/5] Text Score ==="
if ls "$RESULT_DIR"/text_score_${MODE}_*.csv 2>/dev/null | grep -q .; then
    echo "[skip] text already done"
else
    $PYTHON -m scripts.text.text_score \
        --mode "$MODE" \
        --image_dirname "$GEN_OUTPUT_DIR/text" \
        --model_names "$MODEL_NAME" \
        --image_grid "2,2" \
        2>&1 | tee "$RESULT_DIR/logs/text.log"
    copy_fresh_scores text
fi

# ── 3. Diversity Score ────────────────────────────────────────────────────────
echo "=== [3/5] Diversity Score ==="
if ls "$RESULT_DIR"/diversity_score_${MODE}_*.csv 2>/dev/null | grep -q .; then
    echo "[skip] diversity already done"
else
    $PYTHON -m scripts.diversity.diversity_score \
        --mode "$MODE" \
        --image_dirname "$GEN_OUTPUT_DIR" \
        --model_names "$MODEL_NAME" \
        --image_grid "2,2" \
        --class_items $DIV_CLASSES \
        2>&1 | tee "$RESULT_DIR/logs/diversity.log"
    copy_fresh_scores diversity
fi

# ── 4. Style Score ────────────────────────────────────────────────────────────
echo "=== [4/5] Style Score ==="
if ls "$RESULT_DIR"/style_score_${MODE}_*.csv 2>/dev/null | grep -q .; then
    echo "[skip] style already done"
else
    $PYTHON -m scripts.style.style_score \
        --mode "$MODE" \
        --image_dirname "$GEN_OUTPUT_DIR/anime" \
        --model_names "$MODEL_NAME" \
        --image_grid "2,2" \
        2>&1 | tee "$RESULT_DIR/logs/style.log"
    copy_fresh_scores style
fi

# ── 5. Reasoning Score ────────────────────────────────────────────────────────
echo "=== [5/5] Reasoning Score ==="
if ls "$RESULT_DIR"/reasoning_score_${MODE}_*.csv 2>/dev/null | grep -q .; then
    echo "[skip] reasoning already done"
else
    $PYTHON -m scripts.reasoning.reasoning_score \
        --mode "$MODE" \
        --image_dirname "$GEN_OUTPUT_DIR/reasoning" \
        --model_names "$MODEL_NAME" \
        --image_grid "2,2" \
        2>&1 | tee "$RESULT_DIR/logs/reasoning.log"
    copy_fresh_scores reasoning
fi

# ── 汇总结果 ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
# Each phase copied only CSVs created by this invocation.  Never sweep the
# shared benchmark results directory here: it may contain stale experiment
# outputs with the same language mode.

echo ""
echo "=== Alignment ==="
grep -h "overall\|Overall\|score" "$RESULT_DIR/logs/alignment.log" | tail -5 || true
echo "=== Text ==="
grep -h "text_score\|ED\|CR\|WAC" "$RESULT_DIR/logs/text.log" | tail -5 || true
echo "=== Diversity ==="
grep -h "diversity\|score" "$RESULT_DIR/logs/diversity.log" | tail -5 || true
echo "=== Style ==="
grep -h "style_score\|score" "$RESULT_DIR/logs/style.log" | tail -5 || true
echo "=== Reasoning ==="
grep -h "reasoning\|score" "$RESULT_DIR/logs/reasoning.log" | tail -5 || true

$PYTHON "$PROJECT_ROOT/eval/gen/oneig/summarize_scores.py" \
    --result-dir "$RESULT_DIR" --mode "$MODE" --model-name "$MODEL_NAME" \
    --output "$RESULT_DIR/metrics.json"

rm -rf "${BENCHMARK_DIR}/tmp_*" 2>/dev/null || true
echo ""
echo "Done."
