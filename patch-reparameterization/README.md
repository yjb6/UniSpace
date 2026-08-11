# Patch Reparameterization

PatchReparam is the visual-tokenizer component of UniSpace. It combines a
pretrained semantic representation with a compact reconstructive component,
then uses the resulting patch space for image reconstruction and flow-matching
generation.

This directory contains the released SigLIP2, DINOv2, and Qwen3Unified
inference and evaluation recipes. Checkpoints and SHA-256 hashes are hosted at
https://huggingface.co/yjb6/UniSpace.

## Released architecture

The public SigLIP2 and DINOv2 checkpoints use the unified, single-backbone
inference classes `SigLIP2Unified` and `DINOv2Unified`. Some retained training
configs use the historical two-backbone SigLIP2 class because that is the source
training lineage; `src/convert_twobackbone_to_unified.py` converts the selected
checkpoint to the public inference form.

Qwen3Unified is the tokenizer used by the unified multimodal model. It combines
semantic features from Qwen3-VL with a learned reconstructive projection and a
ViT decoder.

## Pipeline

| Stage | Entry point | Purpose |
|---|---|---|
| Tokenizer | `run_stage1.sh` | Train PatchReparam encoder/decoder |
| Qwen3 tokenizer | `run_stage1_qwen3.sh` | Train Qwen3Unified variants |
| Statistics | `run_calculate_stat.sh` | Compute latent normalization statistics |
| DiT | `run_dit.sh` | Train the flow-matching generator |
| Reconstruction | `run_eval_only.sh` | Compute reconstruction metrics |
| Generation | `run_sample_dit.sh` | Distributed sampling and FID |

The selected SigLIP2 and DINOv2 lines use three tokenizer phases: joint
encoder/decoder training, decoder-only GAN finetuning, and decoder-only
finetuning with DiT-generated latent augmentation. The encoder is frozen after
the first phase, so the DiT latent space remains fixed.

## Environment

Use the repository-level `rae` environment. The launchers accept either an
environment name or an environment path through `CONDA_ENV` and default to
`rae`.

The same environment also contains TensorFlow for the canonical OpenAI ADM
FID/sFID/Inception Score/precision/recall evaluator; no second metric-only
environment is required.

```bash
conda env create -f ../environment.yml
conda activate rae
```

Configure external resources explicitly:

| Variable | Purpose |
|---|---|
| `DATA_ROOT` | ImageNet or training dataset root |
| `MODEL_ROOT` | pretrained SigLIP2, DINOv2, Qwen3-VL, and scorer weights |
| `PR_CHECKPOINT_ROOT` | release tokenizer and DiT checkpoints |
| `PR_STATS_ROOT` | release tokenizer normalization statistics |
| `IMAGENET_ROOT` | ImageNet-1K directory containing `train/` and `val/` |
| `IMAGENET_VAL_NPZ` | ImageNet validation reference NPZ |
| `FID_STATS_ROOT` | ADM evaluator directory containing `evaluator.py`, the frozen Inception graph, and the ImageNet reference NPZ |
| `REPO_ROOT` | absolute monorepo root |
| `CONDA_ENV` | `rae` environment name or path |

Configs use OmegaConf environment interpolation and contain public placeholder
defaults rather than internal filesystem paths.

## Commands

Run commands from this directory. Representative retained recipes live under
`configs/stage3/`, `configs/stage1/training/`, and
`configs/stage2/training/ImageNet256/`.

Checkpoint-compatible release evaluation recipes live under
`configs/release/`. PR-SigLIP2 and PR-DINOv2 support both reconstruction and
ImageNet class-conditional generation; PR-Qwen-ViT is released for
reconstruction and multimodal understanding.

```bash
# Tokenizer
bash run_stage1.sh EXPERIMENT configs/stage3/RECIPE.yaml

# Normalization statistics
bash run_calculate_stat.sh

# DiT
bash run_dit.sh EXPERIMENT configs/stage2/training/ImageNet256/RECIPE.yaml

# Reconstruction (replace siglip2 with dinov2 as needed)
bash run_eval_only.sh \
  configs/release/pr-siglip2-imagenet256.yaml \
  "$PR_CHECKPOINT_ROOT/pr-siglip2-tokenizer.pt" \
  --output-dir outputs/pr-siglip2-reconstruction \
  --num-samples 50000 --no-zeroshot

# PR-Qwen-ViT reconstruction
bash run_eval_only.sh \
  configs/release/pr-qwen-vit-imagenet256.yaml \
  "$PR_CHECKPOINT_ROOT/pr-qwen-vit-tokenizer.pt" \
  --output-dir outputs/pr-qwen-vit-reconstruction \
  --num-samples 50000 --no-zeroshot

# ImageNet generation without classifier-free guidance
bash run_sample_dit.sh \
  configs/release/pr-siglip2-imagenet256.yaml \
  outputs/pr-siglip2-imagenet50k \
  --cfg-scale 1.0 --precomputed-latents-dir none \
  --num-fid-samples 50000

# Paper CFG setting (replace siglip2 with dinov2 as needed)
bash run_sample_dit.sh \
  configs/release/pr-siglip2-imagenet256.yaml \
  outputs/pr-siglip2-imagenet50k-cfg1.2 \
  --cfg-scale 1.2 --precomputed-latents-dir none \
  --num-fid-samples 50000
```

Several historical launchers expose additional environment variables for
multi-node execution. Run them with no arguments to see their required inputs
before starting a job.

## Code layout

```text
src/stage1/       PatchReparam orchestration, encoders, and decoders
src/stage2/       DiT and flow-matching transport
src/disc/         discriminator and perceptual losses
src/eval/         reconstruction and generation metrics
src/train_stage1.py
src/train.py
src/sample_ddp.py
src/calculate_stat.py
```

## Release limitations

- Large weights and normalization statistics are distributed through Hugging
  Face rather than bundled in Git.
- This source release is inference/evaluation only; training launchers and
  internal data pipelines are not included.
- Training datasets must be obtained under their respective licenses.
