#!/usr/bin/env bash
# Generate all paper benchmark images with the UniSpace release checkpoint.
# Scoring is intentionally separate because each official benchmark has its
# own environment and, for judge-based metrics, credentials.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UNISPACE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd -- "${UNISPACE_ROOT}/.." && pwd)

: "${UNISPACE_MODEL_PATH:?Set UNISPACE_MODEL_PATH to the SFT checkpoint directory}"
: "${UNISPACE_LLM_PATH:?Set UNISPACE_LLM_PATH to Qwen3-8B}"
: "${UNISPACE_VAE_CONFIG:?Set UNISPACE_VAE_CONFIG to the PR-Qwen-ViT eval YAML}"

OUTPUT_ROOT=${UNISPACE_EVAL_OUTPUT:-"${REPO_ROOT}/eval-results/release"}
BENCHMARKS=${UNISPACE_BENCHMARKS:-"geneval,dpg,imgedit,gedit,oneig_en,oneig_zh"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export PYTHONPATH="${UNISPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch_sdpa}
export USE_FLEX_ATTENTION=${USE_FLEX_ATTENTION:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export UNIMM_GEN_MODULE=eval.gen.gen_images_qwen3_unified_mot

mkdir -p "${OUTPUT_ROOT}"
cd "${UNISPACE_ROOT}"

COMMON=(
  --model-path "${UNISPACE_MODEL_PATH}"
  --llm-path "${UNISPACE_LLM_PATH}"
  --vae-path "${UNISPACE_VAE_CONFIG}"
  --vae-type qwen3_unified
  --patch-reparam-root "${REPO_ROOT}/patch-reparameterization"
  --max_latent_size 96
  --share_unified2llm False
  --use_spatial_merge_und True
  --resolution 1024
  --timestep-shift 0.112
  --distributed
)

selected() {
  [[ ",${BENCHMARKS}," == *",$1,"* ]]
}

run_distributed() {
  local port=$1
  shift
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${port}" "$@"
}

if selected geneval; then
  run_distributed 29601 -m eval.gen.gen_images_geneval \
    --output_dir "${OUTPUT_ROOT}/geneval/images" \
    --metadata_file eval/gen/geneval/prompts/evaluation_metadata.jsonl \
    --num_images_per_prompt 4 --batch_size 4 --cfg_scale 10 \
    --inference-steps 50 "${COMMON[@]}"
fi

if selected dpg; then
  run_distributed 29602 -m eval.gen.gen_images_dpg \
    --output_dir "${OUTPUT_ROOT}/dpg/images" \
    --prompts_folder eval/gen/dpg/dpg_bench/prompts \
    --num_images 4 --batch_size 4 --cfg_scale 10 \
    --inference-steps 50 "${COMMON[@]}"
fi

if selected imgedit; then
  : "${IMGEDIT_BENCH_ROOT:?Set IMGEDIT_BENCH_ROOT for ImgEdit generation}"
  run_distributed 29603 -m eval.gen.gen_images_imgedit_qwen3_unified_mot \
    --output_dir "${OUTPUT_ROOT}/imgedit/images" \
    --metadata_file "${IMGEDIT_BENCH_ROOT}/singleturn.json" \
    --origin_img_root "${IMGEDIT_BENCH_ROOT}" \
    --cfg_text_scale 10 --inference_steps 50 \
    --match-ref-size --ref-use-mar "${COMMON[@]}"
fi

if selected gedit; then
  : "${GEDIT_BENCH_ROOT:?Set GEDIT_BENCH_ROOT for GEdit generation}"
  run_distributed 29604 -m eval.gen.gen_images_gedit_qwen3_unified_mot \
    --output_dir "${OUTPUT_ROOT}/gedit/images" \
    --dataset-path "${GEDIT_BENCH_ROOT}" --language all \
    --cfg_text_scale 10 --inference_steps 50 \
    --match-ref-size --ref-use-mar "${COMMON[@]}"
fi

for mode in EN ZH; do
  bench="oneig_${mode,,}"
  if selected "${bench}"; then
    : "${ONEIG_BENCH_ROOT:?Set ONEIG_BENCH_ROOT for OneIG generation}"
    port=29605
    [[ "${mode}" == ZH ]] && port=29606
    run_distributed "${port}" -m eval.gen.gen_images_oneig_qwen3_unified_mot \
      --mode "${mode}" --benchmark-dir "${ONEIG_BENCH_ROOT}" \
      --output_dir "${OUTPUT_ROOT}/${bench}/images" --model-name unispace \
      --grid-rows 2 --grid-cols 2 --batch_size 4 --cfg_scale 10 \
      --inference_steps 50 "${COMMON[@]}"
  fi
done

printf 'Generation complete: %s\n' "${OUTPUT_ROOT}"
