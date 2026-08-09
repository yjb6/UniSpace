# UniSpace

Official repository for **UniSpace: Patch-Reparameterized Vision Encoders for Unified Representation and Scalable Multimodal Modeling**.

> **Paper preview.** The project page and paper results are available now. Training code, inference code, evaluation recipes, and model checkpoints will be released here shortly.

## Overview

UniSpace builds a unified visual representation for understanding, image generation, and image editing. Its patch-reparameterized vision encoders retain the semantic prior of pretrained visual backbones while learning the image detail needed for reconstruction and generation.

## Results

### Patch-reparameterized visual encoders

| Encoder | PSNR ↑ | SSIM ↑ | rFID ↓ |
|---|---:|---:|---:|
| PR-SigLIP2 | 29.64 | 0.87 | 0.18 |
| PR-DINOv2 | 30.84 | 0.90 | 0.14 |
| PR-Qwen-ViT | 30.16 | 0.88 | 0.17 |

| Encoder | CFG | gFID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| PR-SigLIP2 | No | 4.42 | 6.66 | 190.4 | 0.72 | 0.65 |
| PR-SigLIP2 | Yes | 2.80 | 6.05 | 248.2 | 0.78 | 0.61 |
| PR-DINOv2 | No | 2.10 | 5.39 | 217.2 | 0.78 | 0.64 |
| PR-DINOv2 | Yes | 1.87 | 4.89 | 274.3 | 0.82 | 0.60 |

### UniSpace

| Benchmark | Breakdown | Scores | Overall ↑ |
|---|---|---|---:|
| GenEval | Single / Two / Count / Color / Position / Attribute | 0.98 / 0.92 / 0.69 / 0.88 / 0.83 / 0.73 | 0.84 |
| DPG-Bench | Global / Entity / Attribute / Relation / Other | 84.80 / 92.26 / 90.00 / 94.97 / 88.80 | 86.49 |
| OneIG EN | Align / Text / Reason / Style / Diversity | 0.860 / 0.937 / 0.311 / 0.467 / 0.233 | 0.561 |
| OneIG ZH | Align / Text / Reason / Style / Diversity | 0.807 / 0.881 / 0.276 / 0.455 / 0.244 | 0.533 |

| Benchmark | Breakdown | Scores | Overall ↑ |
|---|---|---|---:|
| ImgEdit | Add / Adjust / Extract / Replace / Remove / Background / Style / Hybrid / Action | 4.53 / 4.38 / 3.61 / 4.67 / 4.42 / 4.23 / 4.55 / 2.70 / 4.47 | 4.28 |
| GEdit EN | Semantic consistency / Perceptual quality | 8.287 / 7.055 | 7.407 |
| GEdit ZH | Semantic consistency / Perceptual quality | 8.270 / 6.998 | 7.382 |

## Release plan

- [x] Paper repository and project page
- [x] Paper metrics and qualitative examples
- [ ] Training and inference code
- [ ] Evaluation and reproduction recipes
- [ ] Model checkpoints

## Citation

Citation metadata will be added with the public paper link.

## License

This repository is released under the license in [LICENSE](LICENSE).
