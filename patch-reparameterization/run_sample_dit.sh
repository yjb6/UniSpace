#!/bin/bash
set -eo pipefail

# 1. 环境激活 (确保使用 Rae 环境)
ENV_PATH="${CONDA_ENV:-rae}"
if [ -f "/conda/bin/activate" ]; then
    source /conda/bin/activate "$ENV_PATH"
else
    source activate "$ENV_PATH"
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR" || { echo "Error: Cannot cd to $WORKDIR"; exit 1; }

CONFIG="${1:?Usage: $0 <config> <sample_dir> [--ckpt PATH] [--cfg-scale SCALE] [--precomputed-latents-dir PATH|none] [--skip-eval] [--num-fid-samples N]}"
SAMPLE_DIR="${2:?Usage: $0 <config> <sample_dir> [--ckpt PATH] [--cfg-scale SCALE] [--precomputed-latents-dir PATH|none] [--skip-eval] [--num-fid-samples N]}"
shift 2
SKIP_EVAL=""
NUM_FID_SAMPLES=""
CKPT_OVERRIDE=""
CFG_SCALE=""
PRECOMPUTED_LATENTS_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-eval) SKIP_EVAL="--skip-eval"; shift ;;
    --num-fid-samples) NUM_FID_SAMPLES="$2"; shift 2 ;;
    --ckpt) CKPT_OVERRIDE="$2"; shift 2 ;;
    --cfg-scale) CFG_SCALE="$2"; shift 2 ;;
    --precomputed-latents-dir) PRECOMPUTED_LATENTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Normalize CONFIG and SAMPLE_DIR to absolute paths to avoid cwd-dependent issues
CONFIG="$(readlink -f "$CONFIG")"
SAMPLE_DIR="$(readlink -f "$SAMPLE_DIR")"

# Paths for later evaluation
EVAL_SCRIPT_DIR="${FID_STATS_ROOT:-/path/to/fid_stats}"
REF_NPZ="${EVAL_SCRIPT_DIR}/VIRTUAL_imagenet256_labeled.npz"
ADM_FID_ENV="${ADM_FID_ENV:-$ENV_PATH}"

export WANDB_MODE=offline
export TORCH_HOME=./cache/torch
export WANDB_DIR=ckpts/stage-dit/$EXPERIMENT_NAME
export HF_HUB_OFFLINE=1
export HF_HOME=./cache/torch
export USE_TORCH=1
export USE_TF=0

# sample_ddp defaults to equal class labels and seed 0, matching the retained
# 50K release runs and their recorded sample_args.json files.
# FID/IS/sFID/precision/recall are computed below with the canonical ADM
# evaluator. Avoid a duplicate in-process FID pass in sample_ddp.
EXTRA_ARGS="--skip-fid"
if [ -n "$NUM_FID_SAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num-fid-samples $NUM_FID_SAMPLES"
fi
if [ -n "$CKPT_OVERRIDE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --ckpt $CKPT_OVERRIDE"
fi
if [ -n "$CFG_SCALE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --cfg-scale $CFG_SCALE"
fi
if [ -n "$PRECOMPUTED_LATENTS_DIR" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --precomputed-latents-dir $PRECOMPUTED_LATENTS_DIR"
fi

if [ "$SKIP_EVAL" != "--skip-eval" ]; then
  if [ ! -f "$REF_NPZ" ]; then
    echo "Reference npz not found: $REF_NPZ; evaluation cannot continue." >&2
    exit 1
  fi
  if [ ! -f "${EVAL_SCRIPT_DIR}/evaluator.py" ] || \
     [ ! -f "${EVAL_SCRIPT_DIR}/classify_image_graph_def.pb" ]; then
    echo "Canonical ADM evaluator assets are incomplete under: $EVAL_SCRIPT_DIR" >&2
    exit 1
  fi
  # Fail before a long 50K sampling run when the unified environment cannot
  # import the TensorFlow runtime required by the canonical ADM evaluator.
  USE_TF=1 USE_TORCH=0 python - <<'PY'
import numpy as np
import tensorflow as tf
print(f"ADM preflight: TensorFlow {tf.__version__}, NumPy {np.__version__}")
PY
fi

torchrun --nproc_per_node="${NUM_GPUS:-8}" --master-port "${MASTER_PORT:-29501}" src/sample_ddp.py \
  --config "$CONFIG" \
  --sample-dir "$SAMPLE_DIR" \
  $EXTRA_ARGS
SAMPLING_EXIT=$?
if [ "$SAMPLING_EXIT" -ne 0 ]; then
  echo "Sampling failed (exit $SAMPLING_EXIT), skip evaluation."
  exit "$SAMPLING_EXIT"
fi

if [ "$SKIP_EVAL" = "--skip-eval" ]; then
  echo "Skip evaluation (--skip-eval)."
  exit 0
fi

# Find the newest .npz under SAMPLE_DIR (sample_ddp writes <sample_folder_dir>.npz there)
SAMPLE_NPZ=$(ls -t "$SAMPLE_DIR"/*.npz 2>/dev/null | head -1)
if [ -z "$SAMPLE_NPZ" ] || [ ! -f "$SAMPLE_NPZ" ]; then
  echo "No .npz found under $SAMPLE_DIR; evaluation cannot continue." >&2
  exit 1
fi

# Normalize to absolute path for evaluator.py
SAMPLE_NPZ="$(readlink -f "$SAMPLE_NPZ")"

EVAL_OUTPUT_FILE="${SAMPLE_DIR}/$(basename "$CONFIG" .yaml)_eval.txt"
echo "Running FID/IS evaluator (ref=$REF_NPZ, sample=$SAMPLE_NPZ), saving to $EVAL_OUTPUT_FILE"

# Run the canonical ADM evaluator in the same rae environment.
set -e
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate $ADM_FID_ENV
source "${EVAL_SCRIPT_DIR}/set_eval_env.sh"
cd "$EVAL_SCRIPT_DIR"
USE_TF=1 USE_TORCH=0 python evaluator.py "$REF_NPZ" "$SAMPLE_NPZ" \
  2>&1 | tee "$EVAL_OUTPUT_FILE"
echo "Evaluation result saved to $EVAL_OUTPUT_FILE"
