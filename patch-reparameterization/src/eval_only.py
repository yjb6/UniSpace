#!/usr/bin/env python3
# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Standalone evaluation script for PatchReparam models (Stage-1 / Stage-2 checkpoints).

Loads a checkpoint, runs reconstruction metrics (PSNR, SSIM, rFID) and optionally
zero-shot ImageNet classification. Supports both single-GPU and multi-GPU via
accelerate launch.

Usage:
    # Single GPU (with checkpoint)
    python eval_only.py --config configs/stage2/xxx.yaml --ckpt path/to/ep-last.pt

    # Pretrained only (no checkpoint, model created from config)
    python eval_only.py --config configs/stage1/pretrained/xxx.yaml --output-dir ./eval_pretrained

    # Multi-GPU
    accelerate launch --num_processes 4 eval_only.py --config configs/stage2/xxx.yaml --ckpt path/to/ep-last.pt

    # Override eval data from config
    python eval_only.py --config xxx.yaml --ckpt xxx.pt --eval-data /path/to/val/ --reference-npz /path/to/val.npz

    # Save reconstructed images (PNG) to eval_results/recon_images/
    python eval_only.py --config xxx.yaml --ckpt xxx.pt --save-recon-images
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from utils.config_utils import load_config  # noqa: F401  (registers oc.env resolver)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from eval import evaluate_reconstruction_distributed, compute_reconstruction_metrics
from eval.clip_zero_shot import (
    IMAGENET_CLASSNAMES,
    CLIP_PAPER_PROMPT_TEMPLATES,
    PatchReparamZeroShotWrapper,
    build_zero_shot_classifier,
    create_imagenet_dataloader_patch_reparam,
    evaluate as evaluate_zero_shot,
)
from stage1 import PatchReparam
from utils.train_utils import center_crop_arr, parse_configs


class ExternalVAEWrapper(torch.nn.Module):
    """Wraps diffusers AutoencoderKL to match the PatchReparam forward interface.

    Input:  images in [0, 1], shape (B, 3, H, W)
    Output: reconstructed images in [0, 1], shape (B, 3, H, W)
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor

    @torch.no_grad()
    def forward(self, images, **kwargs):
        # Disable autocast: Flux VAE requires fp32 precision for accurate reconstruction
        with torch.amp.autocast("cuda", enabled=False):
            images = images.float()
            # AutoencoderKL expects [-1, 1]
            z = self.vae.encode(images * 2 - 1).latent_dist.sample() * self.scaling_factor
            decoded = self.vae.decode(z / self.scaling_factor).sample
            return (decoded + 1) / 2  # back to [0, 1]

class Qwen3VarlenWrapper(torch.nn.Module):
    """Wraps Qwen3Unified PatchReparam to match the standard model(images) eval interface.

    Input:  images in [0, 1], shape (B, 3, 256, 256)
    Output: reconstructed images in [0, 1], shape (B, 3, 256, 256)
    """

    def __init__(self, pr):
        super().__init__()
        self.pr = pr

    def forward(self, images, **kwargs):
        # images: [B, 3, H, W] in [0, 1]
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        p  = self.pr.decoder.config.patch_size
        tp = int(self.pr.encoder.temporal_patch_size)
        H_grid, W_grid = H // p, W // p

        # 从模型取 normalize 参数，不 hardcode
        enc_mean = self.pr.encoder_mean.to(images.device)  # [1, 3, 1, 1]
        enc_std  = self.pr.encoder_std.to(images.device)   # [1, 3, 1, 1]
        frames = images.unsqueeze(1).expand(-1, tp, -1, -1, -1).contiguous()  # [B, tp, 3, H, W]
        frames = (frames - enc_mean.unsqueeze(1)) / enc_std.unsqueeze(1)
        frames = frames.reshape(B, tp, 3, H_grid, p, W_grid, p)
        pixel_format = getattr(self.pr.encoder, 'pixel_format', 't_major')
        if pixel_format == 'c_major':
            # C-major: [B, H_g, W_g, C, tp, p, p] → [B*N, C*tp*p*p]
            frames = frames.permute(0, 3, 5, 2, 1, 4, 6).contiguous().reshape(B * H_grid * W_grid, 3 * tp * p * p)
        else:
            # T-major（旧行为，兼容已有 checkpoint）
            frames = frames.permute(0, 3, 5, 1, 2, 4, 6).contiguous().reshape(B * H_grid * W_grid * tp, 3, p, p)
        grid_thw = torch.tensor([[1, H_grid, W_grid]], dtype=torch.long, device=images.device).expand(B, -1)

        x_recs = self.pr(varlen=True, pixel_values=frames, grid_thw=grid_thw)
        # fast path (all same res) returns tensor [B,3,H,W]; slow path returns list
        if isinstance(x_recs, list):
            x_recs = torch.stack(x_recs)
        return x_recs.clamp(0, 1)


try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    class LoraConfig:
        pass
    class TaskType:
        FEATURE_EXTRACTION = None


def safe_load_checkpoint(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pr_from_checkpoint(
    pr_config,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> PatchReparam:
    """Build PatchReparam from config and load weights from checkpoint."""
    from utils.model_utils import get_obj_from_str

    cfg = OmegaConf.to_container(pr_config, resolve=True) if hasattr(pr_config, "_metadata") else dict(pr_config)
    target = cfg.get("target")
    params = cfg.get("params", {})
    if not target:
        raise KeyError("Config must have 'target' for PatchReparam instantiation.")

    pr = get_obj_from_str(target)(**params)
    pr = pr.to(device).eval()
    pr.requires_grad_(False)

    ckpt = safe_load_checkpoint(ckpt_path, map_location="cpu")
    use_lora = ckpt.get("use_lora", False)
    lora_config = ckpt.get("lora_config", None)

    if use_lora and lora_config and PEFT_AVAILABLE:
        if isinstance(lora_config, dict) and "default" in lora_config:
            lora_config = lora_config["default"]
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(lora_config.get("r", 8)),
                lora_alpha=int(lora_config.get("lora_alpha", 16)),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"]),
                lora_dropout=float(lora_config.get("lora_dropout", 0.1)),
            )
        pr.encoder = get_peft_model(pr.encoder, lora_config)
        logging.info("Applied LoRA adapter from checkpoint.")

    state_dict = ckpt.get("ema" if use_ema else "model")
    if state_dict is None:
        raise KeyError(f"Checkpoint must contain 'ema' or 'model'. Got keys: {list(ckpt.keys())}")

    # Handle DDP/compile prefixes
    clean_state = {}
    for k, v in state_dict.items():
        clean_k = k
        for prefix in ["module._orig_mod.", "_orig_mod.", "module."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                break
        clean_state[clean_k] = v

    missing, unexpected = pr.load_state_dict(clean_state, strict=False)
    if missing:
        logging.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    logging.info(f"Loaded {'EMA' if use_ema else 'model'} weights from {ckpt_path}")
    return pr


def create_pr_from_config(pr_config, device: torch.device) -> PatchReparam:
    """Build PatchReparam from config only (pretrained weights, no checkpoint loading)."""
    from utils.model_utils import get_obj_from_str

    cfg = OmegaConf.to_container(pr_config, resolve=True) if hasattr(pr_config, "_metadata") else dict(pr_config)
    target = cfg.get("target")
    params = cfg.get("params", {})
    if not target:
        raise KeyError("Config must have 'target' for PatchReparam instantiation.")

    pr = get_obj_from_str(target)(**params)
    pr = pr.to(device).eval()
    pr.requires_grad_(False)
    logging.info("Created PatchReparam from config (pretrained, no checkpoint).")
    return pr


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchReparam checkpoint (reconstruction + optional zero-shot).")
    parser.add_argument("--config", type=str, required=True, help="YAML config with stage_1 section.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt). If omitted, model is created from config only (pretrained).")
    parser.add_argument("--eval-data", type=str, default=None, help="Eval dataset path (ImageFolder). Overrides config.")
    parser.add_argument("--reference-npz", type=str, default=None, help="Reference NPZ for metrics. Overrides config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Experiment dir (parent of checkpoints). Default: ckpt's parent's parent.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size for center crop.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU for reconstruction.")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics: psnr, ssim, rfid. Default from config.")
    parser.add_argument("--eval-weights", type=str, choices=["both", "ema_only", "model_only"], default="ema_only",
        help="Which weights to eval: both (default), ema_only, model_only. Overrides config eval.eval_weights.")
    parser.add_argument("--no-zeroshot", action="store_true", help="Disable zero-shot eval even if enabled in config.")
    parser.add_argument("--save-recon-images", action="store_true", help="Save reconstructed images as PNG files to eval_results/recon_images/.")
    parser.add_argument("--image-features-mode", type=str, default=None,
        help="For SigLIP2 encoder: which tokens to use for reconstruction. Overrides config eval.eval_modes when set.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--encoder-cos-sim-model", type=str, default=None,
        help="HuggingFace model path for 'cos_sim' metric. Overrides config eval.encoder_cos_sim_model.")
    parser.add_argument("--psnr-resize", type=int, default=None,
        help="If set, resize images to this resolution (square) before computing PSNR. Overrides config eval.psnr_resize.")
    parser.add_argument("--l1-resize", type=int, default=None,
        help="If set, resize images to this resolution (square) before computing L1. Overrides config eval.l1_resize.")
    # Denoise augmentation: encode -> noise -> DiT denoise -> decode
    parser.add_argument("--denoise-augment", action="store_true",
        help="Enable denoise augmentation: encode GT, add flow-matching noise, DiT denoise, then decode.")
    parser.add_argument("--dit-config", type=str, default=None,
        help="DiT sampling config YAML (must contain stage_2, transport, misc sections).")
    parser.add_argument("--denoise-t", type=float, default=0.1,
        help="Noise level t for flow-matching interpolation (0~1). Default 0.1.")
    parser.add_argument("--denoise-steps", type=int, default=1,
        help="Number of euler steps from t to 0. Default 1.")
    parser.add_argument("--denoise-t-list", type=str, default=None,
        help="Comma-separated t schedule (e.g. '0.8,0.67,0.5,0.0'). Overrides --denoise-t/--denoise-steps.")
    parser.add_argument("--denoise-shift", type=float, default=None,
        help="Time shift value. When set, auto-compute t schedule from shifted 50-step euler, taking last --denoise-steps steps from t_start.")
    parser.add_argument("--denoise-shift-total-steps", type=int, default=50,
        help="Total euler steps for shift schedule computation. Default 50.")
    parser.add_argument("--save-denoise-latents", action="store_true",
        help="Save denoised latents (.pt) and decoded images (.png) under output-dir/denoise_latents/.")
    parser.add_argument("--num-samples", type=int, default=None,
        help="Limit number of eval samples. Default: use all.")
    # Recon gFID: reconstruct train images per-class, compute gFID vs precomputed stats
    parser.add_argument("--recon-gfid", action="store_true",
        help="Reconstruct train images (per-class sampling) and compute gFID.")
    parser.add_argument("--train-data", type=str,
        default=os.environ.get("DATA_ROOT", "/path/to/ImageNet1k") + "/train/",
        help="Train dataset path (ImageFolder) for --recon-gfid.")
    parser.add_argument("--gfid-npz", type=str,
        default=os.path.join(os.environ.get("FID_STATS_ROOT", "/path/to/fid_stats"), "VIRTUAL_imagenet256_labeled.npz"),
        help="Precomputed inception stats NPZ (keys: mu, sigma) for gFID.")
    parser.add_argument("--samples-per-class", type=int, default=50,
        help="Number of samples per class for --recon-gfid. Default 50.")
    return parser.parse_args()


def sample_per_class_indices(dataset, samples_per_class):
    """Return indices for uniform per-class sampling from an ImageFolder dataset."""
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)
    selected = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        if len(indices) >= samples_per_class:
            selected.extend(indices[:samples_per_class])
        else:
            selected.extend(indices)
    return selected


@torch.no_grad()
def evaluate_recon_gfid(
    model,
    dataset,
    gfid_npz_path,
    batch_size,
    autocast_kwargs,
    accelerator,
    experiment_dir=None,
    num_samples=None,
    image_features_mode=None,
    save_recon_dir=None,
):
    """Reconstruct train images and compute gFID vs precomputed inception stats."""
    from eval.fid import calculate_gfid
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    is_main = accelerator.is_main_process

    model.eval()
    N = num_samples if num_samples else len(dataset)
    chunk = N // world_size
    start = rank * chunk
    end = (rank + 1) * chunk if rank < world_size - 1 else N

    subset = Subset(dataset, list(range(start, end)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    reconstructions = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Recon gFID", file=sys.stdout) if is_main else loader

    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                if image_features_mode is not None:
                    recon = model(images, image_features_mode=image_features_mode)
                else:
                    recon = model(images)
            if hasattr(recon, 'x_rec'):
                recon = recon.x_rec
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for img in recon_np:
                reconstructions.append(img)

    reconstructions = np.stack(reconstructions)

    # Gather across ranks
    temp_dir = os.path.join(experiment_dir or ".", "eval_npzs_gfid")
    if is_main:
        os.makedirs(temp_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    shard_path = os.path.join(temp_dir, f"recon_gfid_{rank:02d}.npz")
    np.savez(shard_path, arr_0=reconstructions)
    accelerator.wait_for_everyone()

    metrics = None
    if is_main:
        all_recons = []
        for r in range(world_size):
            shard = np.load(os.path.join(temp_dir, f"recon_gfid_{r:02d}.npz"))["arr_0"]
            all_recons.append(shard)
        combined = np.concatenate(all_recons, axis=0)[:N]

        ref_stats = np.load(gfid_npz_path)
        print(f"[ReconGFID] Recon shape: {combined.shape}")

        # Save reconstructed images in background thread (non-blocking, before metrics)
        if save_recon_dir:
            import threading
            recons_copy = np.copy(combined)
            def _save_images():
                from PIL import Image as _Image
                os.makedirs(save_recon_dir, exist_ok=True)
                for i, img in enumerate(recons_copy):
                    _Image.fromarray(img).save(os.path.join(save_recon_dir, f"recon_{i:06d}.png"))
                print(f"[ReconGFID] Saved {len(recons_copy)} images to {save_recon_dir}")
            t = threading.Thread(target=_save_images, daemon=False)
            t.start()
            print(f"[ReconGFID] Saving {len(combined)} images to {save_recon_dir} in background...")

        device_str = "cuda" if device.type == "cuda" else "cpu"
        gfid = calculate_gfid(combined, ref_stats, batch_size=128, device=device_str)
        metrics = {"gfid": float(gfid)}
        print(f"[ReconGFID] gFID: {gfid:.4f}")

        # cleanup
        for r in range(world_size):
            p = os.path.join(temp_dir, f"recon_gfid_{r:02d}.npz")
            if os.path.exists(p):
                os.remove(p)

    accelerator.wait_for_everyone()
    return metrics


def compute_denoise_t_list(t_start, denoise_steps, shift, shift_total_steps=50):
    """Compute shifted t schedule from t_start to 0, taking last k steps from full schedule."""
    t_uniform = 1 - torch.linspace(0, 1, shift_total_steps + 1)
    t_shifted = shift * t_uniform / (1 + (shift - 1) * t_uniform)
    # Find first index where t_shifted <= t_start
    below = (t_shifted <= t_start).nonzero(as_tuple=True)[0]
    if len(below) == 0:
        return torch.tensor([t_start, 0.0])
    start_idx = below[0].item()
    # Take denoise_steps steps from start_idx, always end at 0
    remaining = t_shifted[start_idx:]
    if denoise_steps >= len(remaining):
        return remaining
    # Evenly sample denoise_steps points from remaining (including endpoints)
    indices = torch.linspace(0, len(remaining) - 1, denoise_steps + 1).long()
    return remaining[indices]


def euler_denoise(x, dit_model, labels, t_start, steps, t_list=None):
    """Euler ODE integration from t_start to 0.

    Args:
        t_list: Optional pre-computed t schedule tensor (including endpoints).
                If provided, t_start and steps are ignored.
    """
    if t_list is not None:
        ts = t_list.to(x.device)
    else:
        ts = torch.linspace(t_start, 0, steps + 1, device=x.device)
    for i in range(len(ts) - 1):
        t_cur = ts[i]
        dt = ts[i + 1] - ts[i]  # negative (going from t to 0)
        t_batch = torch.full((x.shape[0],), t_cur, device=x.device)
        v = dit_model(x, t_batch, y=labels)
        x = x + v * dt
    return x


@torch.no_grad()
def evaluate_denoise_reconstruction(
    pr,
    dit_model,
    dataset,
    denoise_t,
    denoise_steps,
    batch_size,
    autocast_kwargs,
    reference_npz_path,
    metrics_to_compute,
    accelerator,
    experiment_dir=None,
    encoder_cos_sim_model=None,
    psnr_resize=None,
    l1_resize=None,
    num_samples=None,
    save_recon_dir=None,
    t_list=None,
    save_latents_dir=None,
):
    """Evaluate reconstruction with denoise augmentation (encode -> noise -> DiT denoise -> decode)."""
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    is_main = accelerator.is_main_process

    pr.eval()
    dit_model.eval()

    N = min(len(dataset), num_samples) if num_samples else len(dataset)
    chunk = N // world_size
    start = rank * chunk
    end = (rank + 1) * chunk if rank < world_size - 1 else N

    subset = Subset(dataset, list(range(start, end)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    reconstructions = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Denoise Recon", file=sys.stdout) if is_main else loader

    with torch.inference_mode():
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                # 1. encode -> normalized latent
                enc = pr.encode(images, return_aux=True)
                z = enc.merged_tokens_norm  # [B, C, H, W]

                # 2. flow-matching noise: x_t = (1-t)*data + t*noise
                noise = torch.randn_like(z)
                x_t = (1 - denoise_t) * z + denoise_t * noise

                # 3. DiT euler denoise with GT label
                z_hat = euler_denoise(x_t, dit_model, labels, denoise_t, denoise_steps, t_list=t_list)

                # 4. decode (z_hat is [B,C,H,W] normalized, decode will denormalize)
                recon = pr.decode(z_hat)

            if hasattr(recon, 'x_rec'):
                recon = recon.x_rec
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save denoised latents, decoded images, and GT images
            if save_latents_dir is not None:
                from PIL import Image as _Image
                latents_sub = os.path.join(save_latents_dir, "latents")
                gt_sub = os.path.join(save_latents_dir, "gt")
                os.makedirs(latents_sub, exist_ok=True)
                os.makedirs(gt_sub, exist_ok=True)
                gt_np = images.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for j in range(z_hat.shape[0]):
                    global_idx = start + len(reconstructions) + j
                    torch.save(z_hat[j].cpu(), os.path.join(latents_sub, f"{global_idx:06d}.pt"))
                    _Image.fromarray(recon_np[j]).save(os.path.join(save_latents_dir, f"{global_idx:06d}.png"))
                    _Image.fromarray(gt_np[j]).save(os.path.join(gt_sub, f"{global_idx:06d}.png"))

            for img in recon_np:
                reconstructions.append(img)

    reconstructions = np.stack(reconstructions)

    # Gather across ranks via temp npz files
    temp_dir = os.path.join(experiment_dir or ".", "eval_npzs_denoise")
    if is_main:
        os.makedirs(temp_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    shard_path = os.path.join(temp_dir, f"denoise_recon_{rank:02d}.npz")
    np.savez(shard_path, arr_0=reconstructions)
    accelerator.wait_for_everyone()

    metrics = None
    if is_main:
        all_recons = []
        for r in range(world_size):
            shard = np.load(os.path.join(temp_dir, f"denoise_recon_{r:02d}.npz"))["arr_0"]
            all_recons.append(shard)
        combined = np.concatenate(all_recons, axis=0)[:N]

        ref_images = np.load(reference_npz_path)["arr_0"][:N]
        print(f"[DenoiseEval] Combined shape: {combined.shape}, ref shape: {ref_images.shape}")

        # Save reconstructed images in background thread (non-blocking, before metrics)
        if save_recon_dir:
            import threading
            recons_copy = np.copy(combined)
            def _save_images():
                from PIL import Image as _Image
                os.makedirs(save_recon_dir, exist_ok=True)
                for i, img in enumerate(recons_copy):
                    _Image.fromarray(img).save(os.path.join(save_recon_dir, f"recon_{i:06d}.png"))
                print(f"[DenoiseEval] Saved {len(recons_copy)} images to {save_recon_dir}")
            t = threading.Thread(target=_save_images, daemon=False)
            t.start()
            print(f"[DenoiseEval] Saving {len(combined)} images to {save_recon_dir} in background...")

        metrics = compute_reconstruction_metrics(
            ref_images, combined, accelerator.device, 128,
            metrics_to_compute=metrics_to_compute,
            encoder_model=encoder_cos_sim_model,
            psnr_resize=psnr_resize,
            l1_resize=l1_resize,
        )
        print(f"[DenoiseEval] denoise_t={denoise_t}, steps={denoise_steps}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        # cleanup
        for r in range(world_size):
            p = os.path.join(temp_dir, f"denoise_recon_{r:02d}.npz")
            if os.path.exists(p):
                os.remove(p)

    accelerator.wait_for_everyone()
    return metrics


def main():
    args = parse_args()

    mixed_precision = "fp16" if args.precision == "fp16" else ("bf16" if args.precision == "bf16" else "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Only rank 0 logs; suppress other ranks to avoid duplicated output
    root_logger = logging.getLogger()
    if accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
    else:
        root_logger.setLevel(logging.CRITICAL + 1)
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
    logger = logging.getLogger(__name__)

    full_cfg = OmegaConf.load(args.config)

    # External VAE mode (e.g. Flux/SDXL VAE): config has vae_decoder instead of stage_1
    vae_decoder_cfg = full_cfg.get("vae_decoder", None)
    use_external_vae = vae_decoder_cfg is not None

    pr_config = None
    if not use_external_vae:
        pr_config, *_ = parse_configs(full_cfg)
        if pr_config is None:
            raise ValueError("Config must contain stage_1 or vae_decoder section.")

    eval_section = full_cfg.get("eval", None)
    eval_data = args.eval_data or (eval_section.get("data_path") if eval_section else None)
    reference_npz = args.reference_npz or (eval_section.get("reference_npz_path") if eval_section else None)
    eval_metrics = args.metrics or (list(eval_section.get("metrics", ["psnr", "ssim", "rfid"])) if eval_section else ["psnr", "ssim", "rfid"])
    encoder_cos_sim_model = args.encoder_cos_sim_model or (eval_section.get("encoder_cos_sim_model") if eval_section else None)
    psnr_resize = args.psnr_resize or (eval_section.get("psnr_resize") if eval_section else None)
    l1_resize = args.l1_resize or (eval_section.get("l1_resize") if eval_section else None)
    eval_weights = args.eval_weights or (eval_section.get("eval_weights", "both") if eval_section else "both")
    if args.image_features_mode is not None:
        eval_modes = [args.image_features_mode]
    else:
        eval_modes = list(eval_section.get("eval_modes", [None]) if eval_section else [None])
    # print(eval_modes)
    if use_external_vae:
        models_to_eval = ["external_vae"]
    elif args.ckpt is None:
        models_to_eval = ["pretrained"]
    else:
        models_to_eval = ["ema", "model"] if eval_weights == "both" else (["ema"] if eval_weights == "ema_only" else ["model"])

    if not eval_data or not reference_npz:
        raise ValueError("Must specify --eval-data and --reference-npz, or ensure config eval.data_path and eval.reference_npz_path are set.")
    if 'cos_sim' in eval_metrics:
        assert encoder_cos_sim_model, (
            "'cos_sim' metric requires --encoder-cos-sim-model or eval.encoder_cos_sim_model in config."
        )

    # exp_dir: same level as checkpoints (e.g. ckpts/stage1/exp_name/); when no ckpt, require output_dir
    if args.ckpt is not None:
        exp_dir = Path(args.output_dir) if args.output_dir else Path(args.ckpt).parent.parent
    else:
        exp_dir = Path(args.output_dir) if args.output_dir else Path("./eval_pretrained")
    output_dir = str(exp_dir)  # for eval_npzs temp files
    eval_results_dir = exp_dir / "eval_results"
    ckpt_version = "external_vae" if use_external_vae else ("pretrained" if args.ckpt is None else Path(args.ckpt).stem)
    ckpt_output_dir = eval_results_dir / ckpt_version
    os.makedirs(ckpt_output_dir, exist_ok=True)

    # Save config and startup args for reproducibility (main process only)
    if accelerator.is_main_process:
        config_save_path = ckpt_output_dir / "eval_config.yaml"
        OmegaConf.save(full_cfg, config_save_path)
        logger.info("Config saved to %s", config_save_path)
        args_dict = vars(args)
        # Ensure JSON-serializable (e.g. None, bool, int, str, list)
        args_save_path = ckpt_output_dir / "eval_args.json"
        with open(args_save_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        logger.info("Startup args saved to %s", args_save_path)

    device = accelerator.device
    eval_dataset = ImageFolder(
        str(eval_data),
        transform=transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]),
    )
    logger.info(f"Eval dataset: {eval_data}, {len(eval_dataset)} images. Models to eval: {models_to_eval} eval_mode:{eval_modes}")

    autocast_kwargs = {
        "enabled": accelerator.mixed_precision != "no",
        "dtype": torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
    }

    do_zeroshot = (
        not args.no_zeroshot
        and eval_section
        and eval_section.get("zero_shot", {}).get("enabled", False)
    )
    zs_cfg = eval_section.get("zero_shot", {}) if eval_section else {}
    zs_data_path = str(zs_cfg.get("data_path") or zs_cfg.get("imagenet_val_path", ""))
    zs_batch_size = int(zs_cfg.get("batch_size", 128))
    zs_num_workers = int(zs_cfg.get("num_workers", 8))

    # ── Load DiT model if denoise augmentation is enabled ──
    dit_model = None
    if args.denoise_augment:
        if not args.dit_config:
            raise ValueError("--dit-config is required when --denoise-augment is set.")
        from utils.model_utils import instantiate_from_config as _instantiate
        dit_cfg = OmegaConf.load(args.dit_config)
        _, dit_model_config, _, _, _, _, _, _ = parse_configs(dit_cfg)
        logger.info("Loading DiT model for denoise augmentation...")
        dit_model = _instantiate(dit_model_config).to(device).eval()
        dit_model.requires_grad_(False)
        logger.info("DiT loaded. denoise_t=%.3f, denoise_steps=%d", args.denoise_t, args.denoise_steps)

    # ── Build per-class sampled train dataset if recon-gfid is enabled ──
    gfid_dataset = None
    if args.recon_gfid:
        full_train = ImageFolder(
            str(args.train_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ]),
        )
        selected_indices = sample_per_class_indices(full_train, args.samples_per_class)
        gfid_dataset = Subset(full_train, selected_indices)
        logger.info(f"Recon gFID: {len(gfid_dataset)} images ({args.samples_per_class}/class) from {args.train_data}")

    evaluated_samples = min(len(eval_dataset), args.num_samples) if args.num_samples else len(eval_dataset)
    results = {
        "ckpt": args.ckpt,
        "config": args.config,
        "evaluation": {
            "dataset_size": len(eval_dataset),
            "requested_samples": args.num_samples,
            "evaluated_samples": evaluated_samples,
        },
    }
    zs_initialized = False

    for model_name in models_to_eval:
        use_ema = model_name == "ema"
        if use_external_vae:
            from diffusers import AutoencoderKL
            vae_path = vae_decoder_cfg.get("path") if hasattr(vae_decoder_cfg, "get") else str(vae_decoder_cfg)
            logger.info("Loading external VAE from %s", vae_path)
            external_vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
            external_vae.requires_grad_(False)
            logger.info("External VAE loaded: scaling_factor=%s", external_vae.config.scaling_factor)
            pr = ExternalVAEWrapper(external_vae)
        elif args.ckpt is None:
            logger.info("Creating pretrained model from config...")
            pr = create_pr_from_config(pr_config, device)
        else:
            logger.info("Loading %s weights...", model_name)
            pr = load_pr_from_checkpoint(pr_config, args.ckpt, device, use_ema=use_ema)

        # Qwen3Unified 用 varlen 接口，需要 wrapper 转成标准 model(images) 接口
        try:
            encoder_cls = OmegaConf.to_container(pr_config, resolve=True).get("params", {}).get("encoder_cls", "")
        except Exception:
            encoder_cls = ""
        if encoder_cls == "Qwen3Unified":
            pr = Qwen3VarlenWrapper(pr)
            logger.info("Wrapped PatchReparam with Qwen3VarlenWrapper for varlen eval.")

        pr = accelerator.prepare(pr)
        eval_mod = accelerator.unwrap_model(pr)

        model_results = {}

        # ── Denoise augmentation mode ──
        if args.denoise_augment and dit_model is not None:
            # Build t_list: priority: --denoise-t-list > --denoise-shift > linspace
            denoise_t_list = None
            if args.denoise_t_list is not None:
                denoise_t_list = torch.tensor([float(x) for x in args.denoise_t_list.split(",")])
                logger.info("Using manual denoise t_list: %s", denoise_t_list.tolist())
            elif args.denoise_shift is not None:
                denoise_t_list = compute_denoise_t_list(
                    args.denoise_t, args.denoise_steps, args.denoise_shift, args.denoise_shift_total_steps
                )
                logger.info("Using shift-computed denoise t_list (shift=%.1f, steps=%d): %s",
                            args.denoise_shift, args.denoise_steps, denoise_t_list.tolist())
            else:
                logger.info("Running denoise augmentation evaluation for %s (t=%.3f, steps=%d)...",
                            model_name, args.denoise_t, args.denoise_steps)
            save_recon_dir = None
            if args.save_recon_images and accelerator.is_main_process:
                save_recon_dir = str(ckpt_output_dir / "recon_images" / f"{model_name}_denoise_t{args.denoise_t}_s{args.denoise_steps}")
            # Use gfid_dataset (per-class train) if --recon-gfid is also set, else eval_dataset
            denoise_dataset = gfid_dataset if (args.recon_gfid and gfid_dataset is not None) else eval_dataset
            denoise_stats = evaluate_denoise_reconstruction(
                pr=eval_mod,
                dit_model=dit_model,
                dataset=denoise_dataset,
                denoise_t=args.denoise_t,
                denoise_steps=args.denoise_steps,
                batch_size=args.batch_size,
                autocast_kwargs=autocast_kwargs,
                reference_npz_path=args.gfid_npz if (args.recon_gfid and gfid_dataset is not None) else reference_npz,
                metrics_to_compute=eval_metrics,
                accelerator=accelerator,
                experiment_dir=output_dir,
                encoder_cos_sim_model=encoder_cos_sim_model,
                psnr_resize=psnr_resize,
                l1_resize=l1_resize,
                num_samples=args.num_samples,
                save_recon_dir=save_recon_dir,
                t_list=denoise_t_list,
                save_latents_dir=os.path.join(output_dir, "denoise_latents") if args.save_denoise_latents else None,
            )
            if denoise_stats:
                model_results[f"denoise_t{args.denoise_t}_s{args.denoise_steps}"] = {
                    k: float(v) for k, v in denoise_stats.items()
                }
            if accelerator.is_main_process and denoise_stats:
                logger.info("%s denoise augment metrics: %s", model_name, denoise_stats)
        elif args.recon_gfid and gfid_dataset is not None:
            # ── Recon gFID mode: reconstruct train images, compute gFID vs precomputed stats ──
            logger.info("Running recon gFID evaluation for %s (%d images)...", model_name, len(gfid_dataset))
            save_recon_dir = None
            if args.save_recon_images and accelerator.is_main_process:
                save_recon_dir = str(ckpt_output_dir / "recon_images" / f"{model_name}_recon_gfid")
            gfid_stats = evaluate_recon_gfid(
                model=eval_mod,
                dataset=gfid_dataset,
                gfid_npz_path=args.gfid_npz,
                batch_size=args.batch_size,
                autocast_kwargs=autocast_kwargs,
                accelerator=accelerator,
                experiment_dir=output_dir,
                num_samples=len(gfid_dataset),
                image_features_mode=eval_modes[0] if eval_modes[0] else None,
                save_recon_dir=save_recon_dir,
            )
            if gfid_stats:
                model_results["recon_gfid"] = {k: float(v) for k, v in gfid_stats.items()}
            if accelerator.is_main_process and gfid_stats:
                logger.info("%s recon gFID metrics: %s", model_name, gfid_stats)
        else:
            # ── Normal reconstruction evaluation ──
            logger.info("Running reconstruction evaluation for %s...", model_name)
            for eval_mode in eval_modes:
                print(eval_mode)
                save_recon_dir = None
                if args.save_recon_images and accelerator.is_main_process:
                    subdir = model_name if eval_mode is None else f"{model_name}_{eval_mode}"
                    save_recon_dir = str(ckpt_output_dir / "recon_images" / subdir)
                eval_stats = evaluate_reconstruction_distributed(
                    eval_mod,
                    eval_dataset,
                    args.num_samples if args.num_samples else len(eval_dataset),
                    batch_size=args.batch_size,
                    experiment_dir=output_dir,
                    global_step=0,
                    autocast_kwargs=autocast_kwargs,
                    reference_npz_path=reference_npz,
                    metrics_to_compute=eval_metrics,
                    accelerator=accelerator,
                    save_recon_dir=save_recon_dir,
                    image_features_mode=eval_mode,
                    encoder_cos_sim_model=encoder_cos_sim_model,
                    psnr_resize=psnr_resize,
                    l1_resize=l1_resize,
                )
                if eval_stats:
                    recon_key = "reconstruction" if eval_mode is None else f"reconstruction_{eval_mode}"
                    model_results[recon_key] = {k: float(v) for k, v in eval_stats.items()}
                if accelerator.is_main_process and eval_stats:
                    mode_tag = "" if eval_mode is None else f" [{eval_mode}]"
                    logger.info("%s reconstruction metrics%s: %s", model_name, mode_tag, eval_stats)

        if do_zeroshot and zs_data_path and accelerator.is_main_process and not use_external_vae:
            enc_cls = eval_mod.encoder.__class__.__name__
            if "siglip" not in enc_cls.lower():
                logger.warning("Zero-shot skipped for %s: encoder %s is not SigLIP-based.", model_name, enc_cls)
            else:
                try:
                    from transformers import AutoProcessor, SiglipModel
                    sp = full_cfg.get("stage_1")
                    params = getattr(sp, "params", None) if sp is not None else None
                    enc_params = getattr(params, "encoder_params", None) if params is not None else None
                    siglip_path = None
                    if enc_params is not None:
                        siglip_path = getattr(enc_params, "model_name", None) or getattr(enc_params, "sementic_model_name", None)
                    if not siglip_path and params is not None:
                        siglip_path = getattr(params, "encoder_config_path", None)
                    siglip_path = str(siglip_path) if siglip_path else ""
                    if siglip_path:
                        if not zs_initialized:
                            logger.info("Loading full SigLIP for zero-shot from %s", siglip_path)
                            zs_full_siglip = SiglipModel.from_pretrained(siglip_path).to(device).eval()
                            zs_processor = AutoProcessor.from_pretrained(siglip_path)
                            zs_classifier = build_zero_shot_classifier(
                                zs_full_siglip,
                                zs_processor,
                                classnames=IMAGENET_CLASSNAMES,
                                templates=CLIP_PAPER_PROMPT_TEMPLATES,
                                num_classes_per_batch=10,
                                device=device,
                                use_tqdm=True,
                            )
                            zs_dataloader = create_imagenet_dataloader_patch_reparam(
                                zs_data_path,
                                eval_mod.encoder_input_size,
                                eval_mod.encoder_mean,
                                eval_mod.encoder_std,
                                batch_size=zs_batch_size,
                                num_workers=zs_num_workers,
                            )
                            zs_initialized = True
                        zs_wrapper = PatchReparamZeroShotWrapper(eval_mod.encoder, zs_full_siglip)
                        zs_wrapper.eval()
                        zs_top1, zs_top5 = evaluate_zero_shot(
                            zs_wrapper,
                            zs_processor,
                            zs_classifier,
                            zs_dataloader,
                            device=device,
                            precision=args.precision,
                        )
                        model_results["zeroshot"] = {"top1": float(zs_top1), "top5": float(zs_top5)}
                        logger.info("%s zero-shot top1: %.2f, top5: %.2f", model_name, zs_top1, zs_top5)
                    else:
                        logger.warning("Zero-shot disabled: no SigLIP path in config.")
                except Exception as e:
                    logger.warning("Zero-shot failed for %s: %s", model_name, e)

        results[model_name] = model_results

    if do_zeroshot and not zs_data_path and accelerator.is_main_process:
        logger.warning("Zero-shot enabled but data_path not set; skipping.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and results:
        result_stem = "external_vae" if use_external_vae else ("pretrained" if args.ckpt is None else Path(args.ckpt).stem)
        result_path = ckpt_output_dir / f"{result_stem}.json"
        suffix = 1
        while result_path.exists():
            result_path = ckpt_output_dir / f"{result_stem}_{suffix}.json"
            suffix += 1
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation results saved to %s", result_path)
    logger.info("Evaluation done.")


if __name__ == "__main__":
    main()
