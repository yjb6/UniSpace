# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained stage-2 model using DDP and
stores results for downstream metrics. For single-device sampling, use sample.py.

When --reference-npz is provided, FID is computed after sampling completes.
Reference NPZ can be either:
  - Precomputed stats (keys: mu, sigma) from Inception features
  - Raw images (key: arr_0) - FID will be computed via torch_fidelity
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import math
from typing import Callable, Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from utils.config_utils import load_config  # noqa: F401  (registers oc.env resolver)
from utils.model_utils import instantiate_from_config
from stage1 import PatchReparam
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs, center_crop_arr
from eval.fid import calculate_gfid, calculate_rfid
from torchvision.datasets import ImageFolder
from torchvision import transforms


def _safe_load_checkpoint(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # For older PyTorch versions without weights_only arg
        return torch.load(path, map_location=map_location)


def _load_patch_reparam_checkpoint(pr: PatchReparam, ckpt_path: str, use_ema: bool = True) -> None:
    """
    Load checkpoint weights into PatchReparam, choosing between EMA and model weights.
    Mirrors the logic in eval_inversion_error._load_patch_reparam_checkpoint.
    """
    logger.info("Loading PatchReparam checkpoint from %s (use_ema=%s)", ckpt_path, use_ema)
    ckpt = _safe_load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("ema" if use_ema else "model")
    if state_dict is None:
        raise KeyError(f"Checkpoint must contain 'ema' or 'model'. Got keys: {list(ckpt.keys())}")

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
        logger.warning("PatchReparam checkpoint missing %d keys (first 5): %s...", len(missing), missing[:5])
    if unexpected:
        logger.warning("PatchReparam checkpoint unexpected %d keys (first 5): %s...", len(unexpected), unexpected[:5])
    logger.info("PatchReparam checkpoint loaded successfully.")


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    logger.info("Saved .npz file to %s [shape=%s].", npz_path, samples.shape)
    return npz_path


def compute_fid_from_samples(
    gen_npz_path: str,
    reference_npz_path: str,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    """
    Compute FID between generated samples and reference.
    Reference NPZ can be:
      - Precomputed stats: keys 'mu', 'sigma' (Inception moments)
      - Raw images: key 'arr_0'
    """
    gen_data = np.load(gen_npz_path)
    gen_arr = gen_data["arr_0"]

    ref_data = np.load(reference_npz_path)
    ref_files = list(ref_data.files)

    device_str = "cuda" if device.type == "cuda" else "cpu"

    if "mu" in ref_files and "sigma" in ref_files:
        fid = calculate_gfid(gen_arr, ref_data, batch_size=batch_size, device=device_str)
    elif "arr_0" in ref_files:
        ref_arr = ref_data["arr_0"]
        fid = calculate_rfid(gen_arr, ref_arr, bs=batch_size, device=device_str)
    else:
        raise KeyError(
            f"Reference NPZ must have 'mu'/'sigma' (precomputed stats) or 'arr_0' (raw images). "
            f"Got keys: {ref_files}"
        )
    return float(fid)

def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
    eval_data: Optional[str] = None,
    image_size: int = 256,
) -> Callable[[int], torch.Tensor]:
    """Create a callable that returns a batch of labels for the given step index.
    When mode='dataset', labels follow val dataset order (same as training eval)."""

    if sampling_mode == "dataset":
        if not eval_data or not os.path.isdir(eval_data):
            raise ValueError(
                "dataset mode requires --eval-data (or config eval.data_path) pointing to ImageFolder val dir."
            )
        dataset = ImageFolder(
            eval_data,
            transform=transforms.Compose([
                transforms.Lambda(lambda x: center_crop_arr(x, image_size)),
                transforms.ToTensor(),
            ]),
        )
        N = min(len(dataset), num_fid_samples)
        world_size = total_samples // samples_needed_this_device
        chunk = total_samples // world_size  # == samples_needed_this_device
        start = rank * chunk
        end = min(start + chunk, total_samples)
        rank_indices = list(range(start, end))
        device_pool = torch.tensor([dataset.targets[i] for i in rank_indices], dtype=torch.long)
        needed = iterations * batch_size
        if len(device_pool) < needed:
            gen = torch.Generator().manual_seed(seed + rank)
            device_pool = torch.cat([
                device_pool,
                torch.randint(0, num_classes, (needed - len(device_pool),), generator=gen),
            ])
        device_pool = device_pool[:needed].view(iterations, batch_size)

        def dataset_sampler(step_idx: int) -> torch.Tensor:
            return device_pool[step_idx].to(device)

        return dataset_sampler

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)

        return random_sampler

    if sampling_mode == "equal":
        if num_fid_samples % num_classes != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) to be divisible by num_classes ({num_classes})."
            )

        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[permutation]

        if total_samples > num_fid_samples:
            tail = torch.randint(0, num_classes, (total_samples - num_fid_samples,), generator=generator)
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        start = rank * samples_needed_this_device
        end = start + samples_needed_this_device
        device_pool = global_pool[start:end]
        device_pool = device_pool.view(iterations, batch_size)

        def equal_sampler(step_idx: int) -> torch.Tensor:
            labels = device_pool[step_idx]
            return labels.to(device)

        return equal_sampler
    raise ValueError(f"Unknown label sampling mode: {sampling_mode}")

def main(args):
    """Run sampling with distributed execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU. Use sample.py for single-device usage.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        logger.info("Starting rank=%s, seed=%s, world_size=%s.", rank, seed, world_size)

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    cfg = OmegaConf.load(args.config)
    if args.ckpt is not None:
        OmegaConf.update(cfg, "stage_2.ckpt", args.ckpt, force_add=True)
    if args.cfg_scale is not None:
        OmegaConf.update(cfg, "guidance.scale", args.cfg_scale, force_add=True)
    if args.precomputed_latents_dir is not None:
        val = None if args.precomputed_latents_dir.lower() == "none" else args.precomputed_latents_dir
        OmegaConf.update(cfg, "misc.precomputed_latents_dir", val, force_add=True)
    pr_config, model_config, transport_config, sampler_config, guidance_config, misc, _, _ = parse_configs(cfg)
    # ── VAE mode: if vae_decoder is specified, use external VAE instead of PatchReparam ──
    vae_decoder_cfg = cfg.get("vae_decoder", None)
    use_external_vae = vae_decoder_cfg is not None

    if not use_external_vae and (pr_config is None or model_config is None):
        raise ValueError("Config must provide both stage_1 and stage_2 entries (or use vae_decoder).")
    if use_external_vae and model_config is None:
        raise ValueError("Config must provide stage_2 section.")

    misc = {} if misc is None else dict(misc)

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        logger.info("Using time_dist_shift=%.4f.", time_dist_shift)

    # ------------------------------------------------------------------
    # Optional DiT latent caching / reuse
    # ------------------------------------------------------------------
    precomputed_latents_dir = None
    save_latents = False
    latents_dir = None
    if misc is not None:
        precomputed_latents_dir = misc.get("precomputed_latents_dir", None)
        save_latents = bool(misc.get("save_latents", False))
        latents_dir = misc.get("latents_dir", None)

    external_vae = None
    vae_scaling_factor = None
    pr = None
    proj_down_weight = None
    proj_norm_mode = None
    proj_norm_mean = None
    proj_norm_std = None

    if use_external_vae:
        from diffusers import AutoencoderKL
        vae_path = vae_decoder_cfg.get("path") if hasattr(vae_decoder_cfg, "get") else str(vae_decoder_cfg)
        logger.info(f"Loading external VAE from {vae_path}")
        external_vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
        external_vae.requires_grad_(False)
        vae_scaling_factor = external_vae.config.scaling_factor
        logger.info(f"External VAE loaded: scaling_factor={vae_scaling_factor}")

        # Optional projection (for high-dim experiment decoding)
        proj_cfg = vae_decoder_cfg.get("projection", None) if hasattr(vae_decoder_cfg, "get") else None
        if proj_cfg is not None:
            proj_cfg = OmegaConf.to_container(proj_cfg, resolve=True) if hasattr(proj_cfg, '_dict') or type(proj_cfg).__name__ == 'DictConfig' else dict(proj_cfg)
            weight_path = proj_cfg.get("weight_path", None)
            if not weight_path or not os.path.exists(weight_path):
                raise FileNotFoundError(f"Projection weight_path required for sampling: {weight_path}")
            proj_up_weight = torch.load(weight_path, map_location="cpu").to(device)
            proj_down_weight = proj_up_weight.T.contiguous()
            logger.info(f"Loaded projection matrix from {weight_path}: {proj_up_weight.shape}")

            proj_norm_mode = proj_cfg.get("norm_mode", "global")
            if proj_norm_mode == "per_dim":
                norm_stat_path = proj_cfg.get("norm_stat_path", None)
                if norm_stat_path and os.path.exists(norm_stat_path):
                    stats = torch.load(norm_stat_path, map_location="cpu")
                    proj_norm_mean = stats["mean"].to(device)
                    proj_norm_std = torch.sqrt(stats["var"].to(device) + 1e-5)
                    logger.info(f"Loaded per-dim norm stats from {norm_stat_path}")
                else:
                    raise FileNotFoundError(f"per_dim norm requires norm_stat_path: {norm_stat_path}")
    else:
        pr: PatchReparam = instantiate_from_config(pr_config).to(device)
        pr_ckpt_cfg = pr_config.get("checkpoint") if hasattr(pr_config, "get") else None
        if pr_ckpt_cfg:
            if isinstance(pr_ckpt_cfg, (str, Path)):
                ckpt_path = str(pr_ckpt_cfg)
                use_ema = True
            else:
                ckpt_path = pr_ckpt_cfg.get("path") or pr_ckpt_cfg.get("ckpt_path")
                use_ema = pr_ckpt_cfg.get("use_ema", True)
            if ckpt_path and str(ckpt_path).strip():
                _load_patch_reparam_checkpoint(pr, str(ckpt_path), use_ema=use_ema)
            else:
                logger.warning("PatchReparam checkpoint config present but path is empty.")
        else:
            logger.info("No PatchReparam checkpoint specified.")
        pr.eval()

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()

    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    sampler = Sampler(transport)

    sampler_config = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config.get("mode", "ODE")
    sampler_params = dict(sampler_config.get("params", {}))
    mode = sampler_mode.upper()
    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guidance_config = {} if guidance_config is None else dict(guidance_config)

    def guidance_value(key: str, default: float):
        if key in guidance_config:
            return guidance_config[key]
        dashed_key = key.replace("_", "-")
        return guidance_config.get(dashed_key, default)

    guidance_scale = guidance_config.get("scale", 1.0)
    guidance_method = guidance_config.get("method", "cfg")
    t_min = guidance_value("t_min", 0.0)
    t_max = guidance_value("t_max", 1.0)

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guid_model_config = guidance_config.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))

    model_target = model_config.get("target", "stage2")
    model_string_name = str(model_target).split(".")[-1]
    ckpt_path = model_config.get("ckpt")
    ckpt_string_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    sampling_method = sampler_params.get("sampling_method", "na")
    num_steps = sampler_params.get("num_steps", sampler_params.get("steps", "na"))
    guidance_tag = f"cfg-{guidance_scale:.2f}"
    base_components = [model_string_name, ckpt_string_name, guidance_tag, f"bs{args.per_proc_batch_size}"]
    logger.info(f"args.per_proc_batch_size{args.per_proc_batch_size}")
    if mode == "ODE":
        detail_components = [mode, str(num_steps), str(sampling_method), args.precision]
    else:
        diffusion_form = sampler_params.get("diffusion_form", "na")
        last_step = sampler_params.get("last_step", "na")
        last_step_size = sampler_params.get("last_step_size", "na")
        detail_components = [mode, str(num_steps), str(sampling_method), str(diffusion_form), str(last_step), str(last_step_size), args.precision]
    folder_name = "-".join(component.replace(os.sep, "-") for component in base_components + detail_components)
    possible_folder_name = os.environ.get('SAVE_FOLDER', None)
    if possible_folder_name:
        sample_folder_dir = os.path.join(args.sample_dir, possible_folder_name)
    else:
        sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    # Parent dir for result files (one level above image folder)
    result_parent_dir = Path(sample_folder_dir).parent
    folder_base = Path(sample_folder_dir).name

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        logger.info("Saving .png samples at %s", sample_folder_dir)

        # Determine directory to read/write DiT latents.
        if latents_dir is None:
            latents_dir = os.path.join(sample_folder_dir, "latents")

        # Save config and startup args for reproducibility (in parent dir, not inside image folder)
        config_save_path = result_parent_dir / f"{folder_base}_sample_config.yaml"
        OmegaConf.save(cfg, config_save_path)
        logger.info("Config saved to %s", config_save_path)
        args_dict = vars(args)
        args_save_path = result_parent_dir / f"{folder_base}_sample_args.json"
        with open(args_save_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        logger.info("Args saved to %s", args_save_path)
    # Broadcast latents_dir from rank 0 so all ranks agree.
    latents_dir_obj = {"latents_dir": latents_dir}
    obj_list = [latents_dir_obj]
    dist.broadcast_object_list(obj_list, src=0)
    latents_dir = obj_list[0]["latents_dir"]
    if save_latents and latents_dir is not None and rank == 0:
        os.makedirs(latents_dir, exist_ok=True)
    dist.barrier()

    # If npz already exists with enough samples, skip sampling to avoid redundant computation
    npz_path = f"{sample_folder_dir}.npz"
    skip_sampling = False
    if rank == 0:
        logger.info("NPZ path: %s, isfile: %s", npz_path, os.path.isfile(npz_path))
        if os.path.isfile(npz_path):
            try:
                data = np.load(npz_path)
                arr = data["arr_0"]
                logger.info("NPZ already exists at %s with %d samples.", npz_path, arr.shape[0])
                if arr.shape[0] >= args.num_fid_samples:
                    skip_sampling = True
                    logger.info("NPZ already exists at %s with %d samples, skipping sampling.", npz_path, arr.shape[0])
            except Exception as e:
                logger.exception("Error loading NPZ: %s", e)
                # Fall back to existing PNG samples if they are complete enough.
                try:
                    existing_files = [
                        name
                        for name in os.listdir(sample_folder_dir)
                        if os.path.isfile(os.path.join(sample_folder_dir, name)) and name.endswith(".png")
                    ]
                    existing_set = set(existing_files)
                    expected_files = {f"{i:06d}.png" for i in range(args.num_fid_samples)}
                    if expected_files.issubset(existing_set):
                        logger.info(
                            "Found %d PNG samples and a broken NPZ. Rebuilding NPZ from existing images and skipping sampling.",
                            len(existing_files),
                        )
                        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
                        skip_sampling = True
                    else:
                        logger.info(
                            "Existing PNG samples are incomplete (%d found, need at least %d). Proceeding to resample.",
                            len(existing_files),
                            args.num_fid_samples,
                        )
                except Exception as e2:
                    logger.exception("Error while checking/rebuilding NPZ from PNGs: %s", e2)
    skip_tensor = torch.tensor(1 if skip_sampling else 0, device=device)
    dist.broadcast(skip_tensor, 0)
    skip_sampling = skip_tensor.item() == 1
    logger.info("Skip sampling (existing images/NPZ): %s", skip_sampling)
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    existing = [name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and name.endswith(".png"))]
    num_samples = len(existing)
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        logger.info("Total number of images that will be sampled: %d", total_samples)
    if total_samples % world_size != 0:
        raise ValueError("Total samples must be divisible by world size.")
    samples_needed_this_gpu = total_samples // world_size
    if samples_needed_this_gpu % n != 0:
        raise ValueError("Per-rank sample count must be divisible by the per-GPU batch size.")
    iterations = samples_needed_this_gpu // n
    # Resume: align to global_batch_size boundary to preserve label consistency
    total = (num_samples // global_batch_size) * global_batch_size
    start_step = total // global_batch_size
    remaining_iterations = iterations - start_step
    if rank == 0 and start_step > 0:
        logger.info("Resuming from step %d (%d images exist, aligned to %d), %d steps remaining.",
                     start_step, num_samples, total, remaining_iterations)
    pbar = tqdm(range(remaining_iterations)) if rank == 0 else range(remaining_iterations)

    eval_data = args.eval_data
    if eval_data is None and cfg is not None:
        eval_section = cfg.get("eval", None)
        if eval_section is not None:
            eval_data = eval_section.get("data_path", None)
    label_sampler = build_label_sampler(
        args.label_sampling,
        num_classes,
        args.num_fid_samples,
        total_samples,
        samples_needed_this_gpu,
        n,
        device,
        rank,
        iterations,
        args.global_seed,
        eval_data=eval_data if args.label_sampling == "dataset" else None,
        image_size=args.image_size,
    )

    # Decide whether to reuse precomputed DiT latents (skip DiT forward) or run DiT.
    use_precomputed_latents = False
    # if precomputed_latents_dir is None:
    #     # If a dedicated precomputed directory is not provided, reuse latents_dir if present.
    #     precomputed_latents_dir = latents_dir
    if (
        not skip_sampling
        and precomputed_latents_dir is not None
        and os.path.isdir(precomputed_latents_dir)
    ):
        if rank == 0:
            logger.info("Using precomputed DiT latents from %s.", precomputed_latents_dir)
        use_precomputed_latents = True
    use_precomputed_tensor = torch.tensor(1 if use_precomputed_latents else 0, device=device)
    dist.broadcast(use_precomputed_tensor, 0)
    use_precomputed_latents = use_precomputed_tensor.item() == 1
    if rank == 0:
        logger.info("Use precomputed DiT latents: %s", use_precomputed_latents)

    using_cfg = guidance_scale > 1.0
    if not skip_sampling:
        for step_idx in pbar:
            actual_step = start_step + step_idx
            with autocast(**autocast_kwargs):
                if use_precomputed_latents:
                    # Load precomputed DiT latents and only run the decoder.
                    latents = []
                    for local_idx in range(n):
                        index = local_idx * world_size + rank + total
                        latent_path = os.path.join(precomputed_latents_dir, f"{index:06d}.pt")
                        if not os.path.isfile(latent_path):
                            raise FileNotFoundError(
                                f"Expected latent file {latent_path} but it does not exist. "
                                "Please ensure precomputed latents cover all required indices."
                            )
                        latent = torch.load(latent_path, map_location=device)
                        latents.append(latent)
                    latents = torch.stack(latents, dim=0).to(device=device)
                else:
                    # Run DiT to generate latents.
                    z = torch.randn(n, *latent_size, device=device)
                    y = label_sampler(actual_step)

                    model_kwargs = dict(y=y)
                    model_fn = model.forward

                    if using_cfg:
                        z = torch.cat([z, z], dim=0)
                        y_null = torch.full((n,), null_label, device=device)
                        y = torch.cat([y, y_null], dim=0)
                        model_kwargs = dict(
                            y=y,
                            cfg_scale=guidance_scale,
                            cfg_interval=(t_min, t_max),
                        )
                        if guidance_method == "autoguidance":
                            if guid_model_forward is None:
                                raise RuntimeError("Guidance model forward is not initialized.")
                            model_kwargs["additional_model_forward"] = guid_model_forward
                            model_fn = model.forward_with_autoguidance
                        else:
                            model_fn = model.forward_with_cfg
                    latents = sample_fn(z, model_fn, **model_kwargs)[-1]
                    if using_cfg:
                        latents, _ = latents.chunk(2, dim=0)

                    # Optionally cache DiT latents for future runs.
                    if save_latents and latents_dir is not None:
                        for local_idx, latent in enumerate(latents):
                            index = local_idx * world_size + rank + total
                            latent_path = os.path.join(latents_dir, f"{index:06d}.pt")
                            torch.save(latent.to("cpu"), latent_path)

                # Decode latents into images.
                if use_external_vae:
                    # Reverse projection if used
                    if proj_down_weight is not None:
                        if proj_norm_mode == "per_dim" and proj_norm_mean is not None:
                            latents = latents * proj_norm_std + proj_norm_mean
                        latents = torch.nn.functional.conv2d(latents, proj_down_weight[:, :, None, None])
                    decoded = external_vae.decode(latents / vae_scaling_factor).sample
                    decoded = ((decoded + 1) / 2).clamp(0, 1)
                else:
                    decoded = pr.decode(latents)
                    mid_x_rec = getattr(decoded, 'mid_x_rec', None)
                    if hasattr(decoded, 'x_rec'):
                        decoded = decoded.x_rec
                    decoded = decoded.clamp(0, 1)
                if mid_x_rec is not None:
                    mid_decoded = mid_x_rec.clamp(0, 1)
                    mid_decoded = (
                        mid_decoded.mul(255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                else:
                    mid_decoded = None
                decoded = (
                    decoded.mul(255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )

            for local_idx, sample in enumerate(decoded):
                index = local_idx * world_size + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
                if mid_decoded is not None:
                    mid_dir = sample_folder_dir + "_mid"
                    os.makedirs(mid_dir, exist_ok=True)
                    Image.fromarray(mid_decoded[local_idx]).save(f"{mid_dir}/{index:06d}.png")

            total += global_batch_size
            dist.barrier()

    dist.barrier()
    if rank == 0:
        if not skip_sampling:
            npz_path = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        # else: npz_path already set and file exists

        reference_npz = args.reference_npz
        if reference_npz is None and cfg is not None:
            eval_section = cfg.get("eval", None)
            if eval_section is not None:
                reference_npz = eval_section.get("reference_npz_path", None)

        if reference_npz is not None and not args.skip_fid:
            logger.info("Computing FID against reference...")
            fid_value = compute_fid_from_samples(
                npz_path,
                reference_npz,
                device,
                batch_size=args.fid_batch_size,
            )
            logger.info("FID: %.6f", fid_value)

            result_path = result_parent_dir / f"{folder_base}_fid_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fid": fid_value,
                        "reference_npz": reference_npz,
                        "num_samples": args.num_fid_samples,
                        "sample_folder": sample_folder_dir,
                    },
                    f,
                    indent=2,
                )
            logger.info("FID result saved to %s", result_path)

        logger.info("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=125)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal", "dataset"],
        default="equal",
        help="Choose how to sample class labels. Use 'dataset' to match training eval (val order).",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to eval ImageFolder (val dir). Required for --label-sampling dataset. Overrides config eval.data_path.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for center crop when using dataset mode.",
    )
    parser.add_argument(
        "--reference-npz",
        type=str,
        default=None,
        help="Path to reference NPZ for FID computation. Can be precomputed stats (mu/sigma) or raw images (arr_0). "
        "Overrides config eval.reference_npz_path if set.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=128,
        help="Batch size for FID computation.",
    )
    parser.add_argument(
        "--skip-fid",
        action="store_true",
        help="Generate PNG/NPZ artifacts without computing FID in this process.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override stage_2.ckpt in config with this checkpoint path.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Override guidance.scale in config.",
    )
    parser.add_argument(
        "--precomputed-latents-dir",
        type=str,
        default=None,
        help="Override misc.precomputed_latents_dir in config. Pass 'none' to disable precomputed latents.",
    )

    args = parser.parse_args()
    main(args)
