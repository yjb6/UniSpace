# UniSpace unified model

This directory contains the Qwen3-8B Mixture-of-Transformers model used for
unified image understanding, generation, and editing. It depends on the visual
tokenizer implementation in the sibling `../patch-reparameterization/`
directory.

The selected model is the final 12K-step SFT checkpoint (`0012000`) and is
distributed at https://huggingface.co/yjb6/UniSpace. This directory contains
the public inference and evaluation subset; UniSpace training code is not part
of this release.

## Layout

```text
data/       dataset implementations, public catalog schema, retained recipes
modeling/   UniSpace MoT, Qwen3/Qwen3-VL components, tokenizer wrappers
train/      FSDP training entry point and GPU/NPU utilities
eval/       generation and benchmark adapters
scripts/    supported command-line launchers
```

## Runtime

Install the environment from the repository root:

```bash
conda env create -f ../environment.yml
conda activate rae
```

The default public inference path uses PyTorch SDPA. Flex attention compilation
is opt-in through `USE_FLEX_ATTENTION=1`.

## Required artifacts

Inference uses three artifacts:

1. Qwen3-8B, the text-only base LLM.
2. The Qwen3Unified visual tokenizer configuration, checkpoint, and
   normalization statistics.
3. The released UniSpace SFT checkpoint directory containing
   `model.safetensors`.

Set the roots used by `eval/gen/eval_sft_0012000.example.json`:

```bash
export REPO_ROOT=/absolute/path/to/unispace
export MODEL_ROOT=/absolute/path/to/base-models
export UNISPACE_CHECKPOINT=/absolute/path/to/UniSpace-SFT
export PR_CHECKPOINT_ROOT=/absolute/path/to/release-checkpoints
export PR_STATS_ROOT=/absolute/path/to/release-statistics
```

Copy the example JSON to a local file and adjust its artifact layout if needed.
The example itself intentionally contains no internal paths.

## Single-GPU smoke test

From this directory:

```bash
python -m eval.run_generation_config \
  --config ../eval.local.json \
  --check-only

bash scripts/eval/run_gen_gpu_qwen3_mot.sh \
  ../eval.local.json \
  "A watercolor painting of a lighthouse during a winter sunrise."
```

The selected inference parameters are:

| Parameter | Value |
|---|---:|
| Resolution | 1024 |
| CFG scale | 10 |
| Inference steps | 50 |
| Timestep shift | 0.112 |
| Maximum latent size | 96 |
| Spatial merge for understanding | enabled |
| Unified-to-LLM sharing | disabled |

Checkpoint loading fails on missing parameters or unknown extra parameters. The
only accepted retained parameter is the unused understanding position embedding
stored by the selected training checkpoint.

## Dataset configuration

Internal dataset paths are not part of the release. Copy
`data/dataset_catalog.example.yaml`, populate it with your datasets, and set:

```bash
export UNISPACE_DATASET_CATALOG=/absolute/path/to/dataset_catalog.yaml
```

The retained configs in `data/configs/` document the S1 → S2 → S2.5 → S3 → SFT
training mixture. They are recipes rather than bundled datasets.

The published release does not claim one-command training reproduction.

## Benchmark adapters

Generation adapters are retained for GenEval, DPG-Bench, ImgEdit, GEdit, and
OneIG. Several scorers require their official benchmark repositories, model
weights, or judge credentials. Those dependencies are deliberately not hidden
behind internal paths. The score entry points accept explicit paths or standard
Hugging Face identifiers:

```bash
# DPG-Bench (generated images, output text file, resolution)
bash eval/gen/dpg/dpg_bench/dist_eval.sh outputs/dpg/images outputs/dpg/result.txt 1024

# GEdit-Bench; use a local save_to_disk directory or its Hugging Face ID.
export WISE_API_KEY=...
export WISE_API_BASE=...
export GEDIT_DATASET_PATH=stepfun-ai/GEdit-Bench
bash eval/gen/gedit/score.sh outputs/gedit/images outputs/gedit/scores

# OneIG-Bench must be cloned as ./OneIG-Benchmark. Model IDs can be replaced
# with local paths through ONEIG_VL_MODEL, ONEIG_SE_MODEL, and ONEIG_LLM2CLIP_*.
bash eval/gen/oneig/score.sh outputs/oneig_en/images outputs/oneig_en/scores EN unimm
```

GEdit writes `metrics.json` with aggregate, English, and Chinese SC/PQ/Overall.
OneIG writes `metrics.json` with Alignment, Text, Reasoning, Style, Diversity,
and their arithmetic-mean Overall score. DPG writes its official L1 category
breakdown and overall score to `result_dpgbench_score.txt`.

## Attribution

Parts of this directory were derived from ByteDance BAGEL. See
`LICENSE-APACHE` and `NOTICE`.
