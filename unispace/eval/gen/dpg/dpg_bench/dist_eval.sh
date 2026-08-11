#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT_PATH=$1
RESULT_PATH=$2
RESOLUTION=$3
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-4}
PORT=${PORT:-29500}

# mkdir -p /home/$USER/.cache/modelscope/hub/models
# cp -rf $CKPT_ROOT/damo /home/$USER/.cache/modelscope/hub/models

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."
# --multi_gpu
accelerate launch --num_machines 1 --num_processes "$PROCESSES" --mixed_precision "fp16" --main_process_port "$PORT" \
  ./dpg_bench/compute_dpg_bench.py \
  --image-root-path "$IMAGE_ROOT_PATH" \
  --res-path "$RESULT_PATH" \
  --resolution "$RESOLUTION" \
  --pic-num "$PIC_NUM" \
  --vqa-model mplug
