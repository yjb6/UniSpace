#!/bin/bash
set -eo pipefail
# Example script for running eval_only.py
#
# Usage:
#   ./run_eval_only.sh CONFIG [CKPT] [OPTIONS...]
#
# CONFIG:  YAML config (required). Uses default if omitted.
# CKPT:    Checkpoint path (.pt). Optional - if omitted, model is created from config only (pretrained).
# OPTIONS: Any optional args passed through to eval_only.py, e.g.:
#   --output-dir DIR        Required when no ckpt (e.g. ./eval_pretrained)
#   --eval-data PATH        Override eval dataset path
#   --reference-npz PATH    Override reference NPZ path
#   --save-recon-images     Save reconstructed images as PNG
#   --no-zeroshot           Disable zero-shot eval
#   --batch-size N          Batch size per GPU
#   --precision fp32|fp16|bf16
#   --num-samples N         Limit number of eval samples
#
# Denoise augmentation (encode -> noise -> DiT denoise -> decode):
#   --denoise-augment --dit-config DIT_CONFIG --denoise-t 0.1 --denoise-steps 1
#   --save-denoise-latents DIR   Save denoised latents + images for ditfinetune
#
# Recon gFID (reconstruct train images per-class, compute gFID):
#   --recon-gfid --train-data TRAIN_DIR --gfid-npz STATS.npz --samples-per-class 50
#
# Examples:
#   # Pretrained only (no ckpt), must specify output-dir
#   ./run_eval_only.sh configs/stage1/pretrained/SigLIP2_my.yaml --output-dir ./eval_pretrained
#
#   # With checkpoint
#   ./run_eval_only.sh configs/stage2/xxx.yaml ckpts/stage1/xxx/checkpoints/ep-last.pt
#
#   # Denoise augmentation
#   ./run_eval_only.sh DIT_CONFIG CKPT --denoise-augment --dit-config DIT_CONFIG --denoise-t 0.1 --denoise-steps 1
#
#   # Recon gFID (--train-data and --gfid-npz have defaults)
#   ./run_eval_only.sh DIT_CONFIG CKPT --recon-gfid --samples-per-class 50
#
# -h / --help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage:
  bash run_eval_only.sh CONFIG [CKPT] [OPTIONS...]

Arguments:
  CONFIG   YAML config with stage_1 section (required)
  CKPT     Checkpoint .pt path (optional, positional, must not start with -)

Options (passed through to eval_only.py):
  --output-dir DIR          Output directory (required when no ckpt)
  --eval-data PATH          Override eval dataset path
  --reference-npz PATH      Override reference NPZ path
  --metrics M [M ...]       Metrics: psnr, ssim, rfid (default from config)
  --save-recon-images       Save reconstructed images as PNG
  --no-zeroshot             Disable zero-shot eval
  --batch-size N            Batch size per GPU (default 64)
  --precision fp32|fp16|bf16 (default bf16)
  --eval-weights ema_only|model_only|both
  --image-size N            Image size for center crop (default 256)
  --num-samples N           Limit number of eval samples (default: all)

Denoise augmentation options:
  --denoise-augment         Enable denoise augmentation mode (encode->noise->DiT denoise->decode)
  --dit-config PATH         DiT config YAML (required with --denoise-augment)
  --denoise-t FLOAT         Flow-matching noise level t (0~1, default 0.1)
  --denoise-steps INT       Euler steps from t to 0 (default 1)
  --denoise-t-list STR      Comma-separated t schedule (e.g. '0.8,0.67,0.0'). Overrides -t/-steps.
  --denoise-shift FLOAT     Time shift value. Auto-compute t schedule from shifted euler.
  --denoise-shift-total-steps INT  Total steps for shift schedule (default 50)
  --save-denoise-latents DIR  Save denoised latents (.pt) and decoded images (.png)

Recon gFID options:
  --recon-gfid              Reconstruct train images per-class, compute gFID vs precomputed stats
  --train-data PATH         Train dataset path (ImageFolder, required with --recon-gfid)
  --gfid-npz PATH           Precomputed inception stats NPZ (required with --recon-gfid)
  --samples-per-class N     Samples per class (default 50)

Examples:
  # Pretrained model (no ckpt)
  bash run_eval_only.sh configs/stage1/pretrained/SigLIP2_my.yaml --output-dir ./eval_pretrained

  # With checkpoint
  bash run_eval_only.sh configs/stage3/xxx.yaml ckpts/stage1/xxx/checkpoints/ep-last.pt

  # With checkpoint + save images + custom metrics
  bash run_eval_only.sh configs/stage3/xxx.yaml ckpts/stage1/xxx/ep-last.pt \
    --save-recon-images --metrics psnr ssim rfid --batch-size 32

  # External VAE (Flux)
  bash run_eval_only.sh configs/stage1/pretrained/flux_vae.yaml \
    --output-dir ./eval_flux_vae --no-zeroshot

  # Denoise augmentation: encode GT -> add noise(t=0.1) -> DiT 1-step denoise -> decode
  bash run_eval_only.sh configs/stage2/training/ImageNet256/DiT-xxx.yaml \
    ckpts/stage1/xxx/checkpoints/ep-last.pt \
    --denoise-augment --dit-config configs/stage2/training/ImageNet256/DiT-xxx.yaml \
    --denoise-t 0.1 --denoise-steps 1 \
    --metrics psnr ssim rfid --num-samples 1000 --save-recon-images

  # Recon gFID: reconstruct 50k train images (50/class), compute gFID
  # --train-data and --gfid-npz have built-in defaults, only override if needed
  bash run_eval_only.sh configs/stage2/training/ImageNet256/DiT-xxx.yaml \
    ckpts/stage1/xxx/checkpoints/ep-last.pt \
    --recon-gfid --samples-per-class 50
HELP
  exit 0
fi

# All release encoders use the repository-level rae environment.
ENV_PATH="${CONDA_ENV:-rae}"

if [ -f "/conda/bin/activate" ]; then
    source /conda/bin/activate "$ENV_PATH"
else
    source activate "$ENV_PATH"
fi

export WANDB_MODE=offline
export TORCH_HOME=./cache/torch
export WANDB_DIR=ckpts/stage1/$EXPERIMENT_NAME
export HF_HUB_OFFLINE=1
export HF_HOME=./cache/torch
# This stage is PyTorch-only. Prevent Transformers from importing the
# TensorFlow backend merely because the unified environment also contains the
# ADM evaluator's TensorFlow runtime.
export USE_TORCH=1
export USE_TF=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200  # 2h，防止 rFID 计算时 rank0 慢导致 timeout

DEFAULT_CONFIG="configs/stage2/siglip2_isortopy_2backbone.yaml"

CONFIG="${1:-$DEFAULT_CONFIG}"
shift || true

# If next arg does not start with -, treat as checkpoint path
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

# Remaining args are passed through to eval_only.py (strip --qwen3)
EXTRA_ARGS=()
for arg in "$@"; do
  [[ "$arg" == "--qwen3" ]] || EXTRA_ARGS+=("$arg")
done

# Build command
CMD_ARGS=(--config "$CONFIG" --precision bf16 --batch-size 64)
# Release configs carry their own eval paths. These optional overrides are kept
# for clusters that expose ImageNet and its reference statistics separately.
[[ -n "${DATA_ROOT:-}" ]] && CMD_ARGS+=(--eval-data "${DATA_ROOT}/val")
[[ -n "${IMAGENET_VAL_NPZ:-}" ]] && CMD_ARGS+=(--reference-npz "$IMAGENET_VAL_NPZ")
[[ -n "$CKPT" ]] && CMD_ARGS+=(--ckpt "$CKPT")
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Single GPU (uncomment to use)
# python src/eval_only.py "${CMD_ARGS[@]}"

# Multi-GPU
accelerate launch --num_processes "${NUM_GPUS:-8}" src/eval_only.py "${CMD_ARGS[@]}"
