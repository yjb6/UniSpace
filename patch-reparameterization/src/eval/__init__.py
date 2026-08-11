from .ref_iqa import calculate_psnr, calculate_lpips, calculate_ssim
from .fid import calculate_rfid, calculate_gfid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from typing import Union, Dict, Optional
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import sys
from functools import lru_cache
@lru_cache(maxsize=4)
def _load_encoder_cached(model_path: str, device_str: str):
    """
    懒加载并缓存 encoder（按 model_path + device 缓存，进程内只加载一次）。
    使用 transformers.AutoModel + AutoProcessor。
    """
    from transformers import AutoModel, AutoProcessor
    print(f"[CosSim] Loading encoder from {model_path} onto {device_str} (cached)...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device_str).eval()
    return model, processor


@torch.no_grad()
def calculate_encoder_cos_sim(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    encoder_model: str,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """
    用指定 encoder 对 GT 图像和重建图像编码，计算 image embedding 的 cos 相似度均值。

    Args:
        ref_arr: GT 图像 [N, H, W, C] uint8
        rec_arr: 重建图像 [N, H, W, C] uint8
        encoder_model: HuggingFace 模型路径或名称
        device: 计算设备
        batch_size: 每批编码的图像数

    Returns:
        所有样本的 cos 相似度均值 (float)
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"
    model, processor = _load_encoder_cached(encoder_model, device_str)

    N = len(ref_arr)
    cos_sims = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ref_batch = [Image.fromarray(img) for img in ref_arr[start:end]]
        rec_batch = [Image.fromarray(img) for img in rec_arr[start:end]]

        # 预处理：使用 processor 处理两组图像
        ref_inputs = processor(images=ref_batch, return_tensors="pt").to(device_str)
        rec_inputs = processor(images=rec_batch, return_tensors="pt").to(device_str)

        # 提取 image embedding
        ref_feats = model.get_image_features(**ref_inputs)  # [B, D]
        rec_feats = model.get_image_features(**rec_inputs)  # [B, D]

        # L2 归一化后计算 cos 相似度
        ref_feats = F.normalize(ref_feats.float(), dim=-1)
        rec_feats = F.normalize(rec_feats.float(), dim=-1)
        batch_cos = (ref_feats * rec_feats).sum(dim=-1)  # [B]
        cos_sims.append(batch_cos.cpu())

    cos_sim_mean = torch.cat(cos_sims).mean().item()
    return cos_sim_mean


def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute = ("psnr", "ssim", "rfid"),
    disable_bar: bool = True,
    encoder_model: Optional[str] = None,
    psnr_resize: Optional[int] = None,
    l1_resize: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute reconstruction metrics between reference and reconstructed images.

    Args:
        ref_arr: Reference images [N, H, W, C] uint8
        rec_arr: Reconstructed images [N, H, W, C] uint8
        device: Device for computation
        batch_size: Batch size for metric computation

    Returns:
        Dictionary with metrics: eval/psnr, eval/ssim, eval/rfid
        Note: LPIPS is not computed here since it's already tracked during training
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"
    # For paired metrics (psnr, ssim, l1, cos_sim), align ref/rec count
    n_paired = min(len(ref_arr), len(rec_arr))
    ref_paired = ref_arr[:n_paired]
    rec_paired = rec_arr[:n_paired]
    results_dict = {}
    if 'psnr' in metrics_to_compute:
        psnr = calculate_psnr(ref_paired, rec_paired, batch_size, device_str, disable_bar=disable_bar, psnr_resize=psnr_resize)
        results_dict["psnr"] = psnr
    if 'ssim' in metrics_to_compute:
        ssim = calculate_ssim(ref_paired, rec_paired, batch_size, device_str, disable_bar=disable_bar)
        results_dict["ssim"] = ssim
    if 'rfid' in metrics_to_compute:
        rfid = calculate_rfid(ref_arr, rec_arr, batch_size, device_str)
        results_dict["rfid"] = rfid
    if 'l1' in metrics_to_compute:
        # L1 = mean(|rec - ref|), same as train: (recon - images).abs().mean()
        ref_f = ref_paired.astype(np.float32) / 255.0
        rec_f = rec_paired.astype(np.float32) / 255.0
        if l1_resize is not None:
            # resize via torch interpolate to keep consistent with psnr_resize style
            # Process in small batches to avoid CUDA OOM on large datasets
            device_str = "cuda" if device.type == "cuda" else "cpu"
            resize_batch = 64
            n = ref_f.shape[0]
            l1_sum = 0.0
            l1_count = 0
            for i in range(0, n, resize_batch):
                ref_b = torch.from_numpy(ref_f[i:i+resize_batch]).permute(0, 3, 1, 2).to(device_str)
                rec_b = torch.from_numpy(rec_f[i:i+resize_batch]).permute(0, 3, 1, 2).to(device_str)
                ref_b = F.interpolate(ref_b, size=(l1_resize, l1_resize), mode="bilinear", align_corners=False)
                rec_b = F.interpolate(rec_b, size=(l1_resize, l1_resize), mode="bilinear", align_corners=False)
                l1_sum += (ref_b - rec_b).abs().sum().item()
                l1_count += ref_b.numel()
            l1 = l1_sum / l1_count
        else:
            l1 = np.abs(rec_f - ref_f).mean()
        results_dict["l1"] = float(l1)
    if 'cos_sim' in metrics_to_compute:
        assert encoder_model is not None, \
            "'cos_sim' metric requires encoder_model (HuggingFace model path) to be specified."
        print(f"[CosSim] Computing encoder cos_sim with model: {encoder_model}")
        cos_sim = calculate_encoder_cos_sim(
            ref_paired, rec_paired,
            encoder_model=encoder_model,
            device=device,
            batch_size=batch_size,
        )
        results_dict["cos_sim"] = cos_sim
    assert len(results_dict) > 0, "No metrics were computed."
    return results_dict
def compute_generation_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
):
    device_str = "cuda" if device.type == "cuda" else "cpu"
    # only eval FID
    fid = calculate_gfid(rec_arr, ref_arr, batch_size, device_str)
    return {
        'fid': fid
    }
@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size, # for noise
    additional_model_kwargs,
    use_guidance: bool,
    pr,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: Union[int, None] = None,
    world_size: Union[int, None] = None,
    device: torch.device = None,
    experiment_dir: str = None,
    global_step: int = None,
    autocast_kwargs: dict = None,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    accelerator: Union[Accelerator, None] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate generation metrics using all GPUs in a distributed manner.

    Args:
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for generation
        rank: Current GPU rank (deprecated, use accelerator instead)
        world_size: Total number of GPUs (deprecated, use accelerator instead)
        device: Device to use (deprecated, use accelerator instead)
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file
        accelerator: Accelerator instance (preferred over rank/world_size/device)

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # Support both Accelerator and legacy DDP interface
    if accelerator is not None:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        wait_for_everyone = accelerator.wait_for_everyone
    else:
        # Legacy DDP mode
        import torch.distributed as dist
        if rank is None or world_size is None or device is None:
            raise ValueError("Either accelerator must be provided, or rank/world_size/device must be provided")
        is_main_process = (rank == 0)
        wait_for_everyone = dist.barrier

    # model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if is_main_process:
        print(f"\n[Eval] Starting distributed sampling evaluation at step {global_step}")
        os.makedirs(temp_dir, exist_ok=True)

    # Wait for rank 0 to create the directory before other ranks try to save
    wait_for_everyone()
    # print(f"[Rank {rank}] Starting sampling...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Reconstruct images on this rank
    generations = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if is_main_process else loader

    with torch.inference_mode():
        for _, label in iterator: # don't actually need images at sampling time
            n = label.size(0)
            z = torch.randn(n, *latent_size, device = device)
            y = label.to(device)
            if use_guidance:
                z = torch.cat([z, z], dim=0)
                y_null = torch.full((n,), null_label, device=device)
                y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, **additional_model_kwargs)
            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples = samples.chunk(2, dim = 0)
                if callable(pr) and not hasattr(pr, 'decode'):
                    samples = pr(samples)
                else:
                    samples = pr.decode(samples)
                if hasattr(samples, 'x_rec'):
                    samples = samples.x_rec
                samples = samples.clamp(0, 1)
            gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for img in gen_np:
                generations.append(img)

    generations = np.stack(generations)
    shard_path = os.path.join(temp_dir, f"gen_{global_step:07d}_{rank:02d}.npz")
    np.savez(shard_path, arr_0=generations)

    if is_main_process:
        print(f"[Rank {rank}] Saved {len(generations)} generation to {shard_path}")

    # Wait for all ranks to finish generation
    wait_for_everyone()

    # Rank 0 computes metrics
    metrics = None
    if is_main_process:
        # Combine all generation shards
        all_gens = []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_gens.append(shard_data)

        combined_recons = np.concatenate(all_gens, axis=0)[:num_samples]
        print(f"[Eval] Combined generation NPZ shape: {combined_recons.shape}")

        # Load reference NPZ
        ref_npz_path = reference_npz_path

        if not os.path.exists(ref_npz_path):
            raise FileNotFoundError(f"Reference NPZ not found at {ref_npz_path}")

        ref_stats = np.load(ref_npz_path)
        print(f"[Eval] Loaded reference NPZ from {ref_npz_path}")

        # Compute metrics
        print("[Eval] Computing metrics...")
        metrics = compute_generation_metrics(
            ref_stats,
            combined_recons,
            device,
            metric_batch_size,
        )

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Cleanup reconstruction shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    wait_for_everyone()
    return metrics
@torch.no_grad()
def evaluate_reconstruction_distributed(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: Union[int, None] = None,
    world_size: Union[int, None] = None,
    device: torch.device = None,
    experiment_dir: str = None,
    global_step: int = None,
    autocast_kwargs: dict = None,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    metrics_to_compute: Optional[list] = ("psnr", "ssim", "rfid"),
    accelerator: Union[Accelerator, None] = None,
    save_recon_dir: Optional[str] = None,
    image_features_mode: Optional[str] = None,
    encoder_cos_sim_model: Optional[str] = None,
    psnr_resize: Optional[int] = None,
    l1_resize: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        model: Model to evaluate (should be in eval mode)
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank (deprecated, use accelerator instead)
        world_size: Total number of GPUs (deprecated, use accelerator instead)
        device: Device to use (deprecated, use accelerator instead)
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file
        accelerator: Accelerator instance (preferred over rank/world_size/device)

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # Support both Accelerator and legacy DDP interface
    if accelerator is not None:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        wait_for_everyone = accelerator.wait_for_everyone
    else:
        # Legacy DDP mode
        import torch.distributed as dist
        if rank is None or world_size is None or device is None:
            raise ValueError("Either accelerator must be provided, or rank/world_size/device must be provided")
        is_main_process = (rank == 0)
        wait_for_everyone = dist.barrier

    model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if is_main_process:
        print(f"\n[Eval] Starting distributed reconstruction evaluation at step {global_step}")
        os.makedirs(temp_dir, exist_ok=True)
    # Wait for rank 0 to create the directory before other ranks try to save
    wait_for_everyone()
    # print(f"[Rank {rank}] Starting reconstruction...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Reconstruct images on this rank, keep originals for distributed metrics
    reconstructions = []
    originals = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Reconstructing", file=sys.stdout) if is_main_process else loader

    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device, non_blocking=True)
            # Save originals before model forward
            orig_np = images.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            with autocast(**autocast_kwargs):
                if image_features_mode is not None:
                    recon = model(images, image_features_mode=image_features_mode)
                else:
                    recon = model(images)

            # Handle DecodeOutput from split_adaln decoder
            if hasattr(recon, 'x_rec'):
                recon = recon.x_rec
            # Convert to numpy uint8 [H, W, C]
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for img in recon_np:
                reconstructions.append(img)
            for img in orig_np:
                originals.append(img)

    reconstructions = np.stack(reconstructions)
    originals = np.stack(originals)

    # Save shard for backward compat
    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{rank:02d}.npz")
    np.savez(shard_path, arr_0=reconstructions)
    if is_main_process:
        print(f"[Rank {rank}] Saved {len(reconstructions)} reconstructions to {shard_path}")

    # Wait for all ranks to finish reconstruction
    wait_for_everyone()

    # ── Distributed metrics: each rank computes on its own subset ──────────
    import torch.distributed as dist
    from .fid import compute_inception_features, calculate_rfid_from_features
    local_n = len(reconstructions)

    # PSNR / SSIM: compute locally then all_reduce weighted sum
    scalar_metrics_to_compute = [m for m in (metrics_to_compute or []) if m in ('psnr', 'ssim', 'lpips')]
    local_scalar = {}
    if scalar_metrics_to_compute:
        if is_main_process:
            print(f"[Eval] Rank {rank}: computing {scalar_metrics_to_compute} on {local_n} images...")
        local_scalar = compute_reconstruction_metrics(
            originals, reconstructions, device, metric_batch_size,
            metrics_to_compute=scalar_metrics_to_compute,
            disable_bar=is_main_process,
            encoder_model=encoder_cos_sim_model,
            psnr_resize=psnr_resize,
            l1_resize=l1_resize,
        )

    # Weighted all_reduce: sum(metric * count) / total_count
    global_scalar = {}
    count_t = torch.tensor(float(local_n), device=device)
    distributed = world_size > 1 and dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    for k, v in local_scalar.items():
        val_t = torch.tensor(v * local_n, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
        global_scalar[k] = (val_t / count_t).item()

    # rFID: each rank computes inception features, all_gather, rank 0 computes FID
    global_rfid = {}
    if 'rfid' in (metrics_to_compute or []):
        if is_main_process:
            print(f"[Eval] Rank {rank}: computing inception features on {local_n} images...")
        local_recon_feats = compute_inception_features(reconstructions, metric_batch_size, device)
        local_ref_feats   = compute_inception_features(originals,       metric_batch_size, device)

        recon_t = torch.from_numpy(local_recon_feats).to(device)
        ref_t   = torch.from_numpy(local_ref_feats).to(device)

        if distributed:
            all_recon = [torch.zeros_like(recon_t) for _ in range(world_size)]
            all_ref   = [torch.zeros_like(ref_t)   for _ in range(world_size)]
            dist.all_gather(all_recon, recon_t)
            dist.all_gather(all_ref, ref_t)
        else:
            all_recon = [recon_t]
            all_ref = [ref_t]

        if is_main_process:
            combined_recon_feats = torch.cat(all_recon, dim=0).cpu().double().numpy()
            combined_ref_feats   = torch.cat(all_ref,   dim=0).cpu().double().numpy()
            rfid = calculate_rfid_from_features(combined_recon_feats, combined_ref_feats)
            global_rfid = {'rfid': rfid}

    metrics = None
    if is_main_process:
        metrics = {**global_scalar, **global_rfid}
        # Persist the all-reduced count so release validation proves how many
        # examples were actually processed rather than trusting CLI intent.
        metrics["_evaluated_samples"] = int(count_t.item())
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Optionally save reconstructed images
        if save_recon_dir:
            import threading
            recons_copy = np.copy(reconstructions)
            save_dir = save_recon_dir
            def _save_recon_images():
                os.makedirs(save_dir, exist_ok=True)
                for i, img in enumerate(recons_copy):
                    Image.fromarray(img).save(os.path.join(save_dir, f"recon_{i:06d}.png"))
                print(f"[Eval] Saved {len(recons_copy)} reconstruction images to {save_dir}")
            threading.Thread(target=_save_recon_images, daemon=False).start()

        # Cleanup shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    wait_for_everyone()
    model.train()

    return metrics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-img", type=str, default="samples/imagenet-256-val.npz")
    parser.add_argument("--rec-img", type=str, default="samples/sdvae-ft-mse-f8d4.npz")
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    # Load images
    device = "cuda"
    ref_img = np.load(args.ref_img)["arr_0"]
    rec_img = np.load(args.rec_img)["arr_0"]
    print(f"Loaded images: ref: {ref_img.shape}, rec: {rec_img.shape}")

    psnr = calculate_psnr(ref_img, rec_img, args.bs, device)
    print(f"PSNR: {psnr:.6f}")
    lpips = calculate_lpips(ref_img, rec_img, args.bs, device)
    print(f"LPIPS: {lpips:.6f}")
    ssim_val = calculate_ssim(ref_img, rec_img, args.bs, device)
    print(f"SSIM: {ssim_val:.6f}")
    rfid = calculate_rfid(ref_img, rec_img, args.bs, device)
    print(f"rFID: {rfid:.6f}")
