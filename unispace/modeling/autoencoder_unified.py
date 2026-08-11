"""PatchReparam (Reconstructive AutoEncoder) wrapper for unimm-transfusion.

Provides the same interface as Flux VAE (autoencoder.py / autoencoder_flux2.py):
  model, config = load_unified_vae(checkpoint_path, unified_vae_config_path)
  config.downsample  -> 16  (patch_size)
  config.z_channels  -> 768/1280 (latent dim)
  model.encode(x)    -> [B, z_channels, H/16, W/16]
  model.decode(z)    -> [B, 3, H, W]

PatchReparam 代码从 patch-reparameterization 项目动态 import，不复制进来，保证版本始终最新。
默认路径通过 _DEFAULT_PATCH_REPARAM_ROOT 指定，也可通过 load_unified_vae(patch_reparam_root=...) 覆盖。
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor

logger = logging.getLogger(__name__)

# Register OmegaConf's oc.env resolver so VAE configs can reference environment
# variables (e.g. ${oc.env:MODEL_ROOT,/path/to/models}/Qwen3-VL-8B-Instruct).
# Safe to call multiple times.
if not OmegaConf.has_resolver("oc.env"):
    OmegaConf.register_new_resolver("oc.env", lambda var, default=None: os.environ.get(var, default))

# patch-reparameterization 子目录的默认根路径（monorepo 内）。
# 优先环境变量 UNISPACE_PATCH_REPARAM_ROOT；否则基于本文件位置推断：
#   <repo_root>/unispace/modeling/autoencoder_unified.py  →  <repo_root>/patch-reparameterization
# 也可通过 load_unified_vae(patch_reparam_root=...) 显式覆盖。
def _default_patch_reparam_root() -> str:
    env = os.environ.get("UNISPACE_PATCH_REPARAM_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))      # .../unispace/modeling
    repo_root = os.path.dirname(os.path.dirname(here))     # .../unispace  (仓库根)
    return os.path.join(repo_root, "patch-reparameterization")


_DEFAULT_PATCH_REPARAM_ROOT = _default_patch_reparam_root()


def _ensure_patch_reparam_importable(patch_reparam_root: Optional[str] = None):
    """把 patch-reparameterization/src 加入 sys.path，保证 stage1.patch_reparam 可以 import。"""
    root = patch_reparam_root or _DEFAULT_PATCH_REPARAM_ROOT
    src = os.path.join(root, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


@dataclass
class UnifiedVAEConfig:
    """Config compatible with AutoEncoderParams / Flux2VAEConfig."""
    downsample: int = 16    # PatchReparam patch_size=16, equivalent 16x spatial downsample
    z_channels: int = 768   # PatchReparam latent dimension


class UnifiedVAEWrapper(nn.Module):
    """Wraps a full PatchReparam model to expose encode/decode with the same
    interface as Flux VAE for seamless integration with unimm-transfusion.

    Input/output range convention matches Flux VAE: [-1, 1].
    Encode output is determined by the PatchReparam config:
      - If do_normalization=True and norm_feature != 'z': returns normalized
        merged_tokens (e.g. merged_tokens_norm) as the flow matching target.
      - Otherwise: returns z (recon_post, the default).
    Decode handles denormalization and recon_proj internally (configured in PatchReparam).
    """

    def __init__(self, unified_vae: nn.Module):
        super().__init__()
        self.unified_vae = unified_vae
        # Determine encode output mode from PatchReparam config
        self._use_merged_tokens_norm = (
            getattr(unified_vae, 'do_normalization', False)
            and getattr(unified_vae, 'norm_feature', 'z') != 'z'
        )
        # Detect if encoder requires varlen (pixel_values + grid_thw) interface
        enc_cls = type(unified_vae.encoder).__name__
        self._use_varlen_encode = enc_cls == 'Qwen3Unified'
        # pixel_format: 'c_major' = [N, C*T*p*p] per patch（与官方 Qwen3VL 一致）
        #               't_major' = [N*T, C, p, p] per patch（旧行为，向后兼容）
        self._pixel_format = getattr(unified_vae.encoder, 'pixel_format', 't_major')
        if self._use_merged_tokens_norm:
            logger.info(f"UnifiedVAEWrapper: encode returns merged_tokens_norm "
                        f"(norm_feature={unified_vae.norm_feature})")
        else:
            logger.info("UnifiedVAEWrapper: encode returns z (recon_post)")
        if self._use_varlen_encode:
            logger.info(f"UnifiedVAEWrapper: using varlen encode path for Qwen3Unified encoder "
                        f"(pixel_format={self._pixel_format})")

    def _image_to_varlen(self, x: Tensor):
        """Convert a single [C, H, W] image to Qwen3Unified varlen format.

        pixel_format='c_major'（推荐）:
            pixel_values: [H/16 * W/16, C*2*16*16]  — C-major per patch，与官方 Qwen3VL 一致
        pixel_format='t_major'（旧行为，向后兼容）:
            pixel_values: [H/16 * W/16 * 2, C, 16, 16]  — T-major per patch

        Returns:
            pixel_values: 见上
            grid_thw:     [1, 3]  with values [1, H//16, W//16]
        """
        C, H, W = x.shape
        p, tp = 16, 2
        h_grid, w_grid = H // p, W // p
        frames = x.unsqueeze(0).expand(tp, -1, -1, -1).contiguous()  # [2, C, H, W]
        frames = frames.reshape(tp, C, h_grid, p, w_grid, p)
        if self._pixel_format == 'c_major':
            # [h_grid, w_grid, C, T, p, p] → [N, C*T*p*p]  C-major per patch
            frames = frames.permute(2, 4, 1, 0, 3, 5)               # [h_g, w_g, C, T, p, p]
            pixel_values = frames.reshape(h_grid * w_grid, C * tp * p * p)
        else:
            # [h_grid, w_grid, T, C, p, p] → [N*T, C, p, p]  T-major per patch（旧行为）
            frames = frames.permute(2, 4, 0, 1, 3, 5)
            pixel_values = frames.reshape(h_grid * w_grid * tp, C, p, p)
        grid_thw = torch.tensor([[1, h_grid, w_grid]], dtype=torch.long, device=x.device)
        return pixel_values, grid_thw

    def _images_to_varlen(self, x: Tensor):
        """Convert [B, C, H, W] same-resolution images to Qwen3Unified varlen format."""
        pv_list, thw_list = [], []
        for i in range(x.shape[0]):
            pv, thw = self._image_to_varlen(x[i])
            pv_list.append(pv)
            thw_list.append(thw)
        return torch.cat(pv_list, dim=0), torch.cat(thw_list, dim=0)

    def encode(self, x: Tensor) -> Tensor:
        """
        Input:  [B, 3, H, W] images in [-1, 1] range (same as Flux VAE interface)
        Output: [B, z_channels, h, w] where h=H/16, w=W/16

        Qwen3Unified path: encoder expects [-1, 1] patches (caller's responsibility to normalize).
        Legacy PatchReparam path: converts [-1,1] → [0,1] before calling PatchReparam.
        """
        if self._use_varlen_encode:
            # Qwen3Unified expects [-1, 1] — no rescaling needed
            B, _, H, W = x.shape
            h_grid, w_grid = H // 16, W // 16
            pixel_values, grid_thw = self._images_to_varlen(x)
            # encode_varlen returns [total_tokens, latent_dim]; reshape to [B, C, h, w]
            z_flat = self.unified_vae.encode_varlen(pixel_values, grid_thw)
            z = z_flat.reshape(B, h_grid * w_grid, -1).permute(0, 2, 1)
            z = z.reshape(B, -1, h_grid, w_grid)
            return z

        x = x * 0.5 + 0.5  # [-1, 1] → [0, 1] for legacy PatchReparam encoder

        if self._use_merged_tokens_norm:
            enc_output = self.unified_vae.encode(x, return_aux=True)
            z = enc_output.merged_tokens_norm
            if z is None:
                raise RuntimeError(
                    "PatchReparam config has do_normalization + norm_feature=merged_tokens, "
                    "but encode returned merged_tokens_norm=None. "
                    "Check normalization_stat_path in PatchReparam config."
                )
            return z
        else:
            z = self.unified_vae.encode(x, return_aux=False)
            return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Input:  [B, z_channels, h, w]
        Output: [B, 3, H, W] images in [-1, 1] range (same as Flux VAE interface)

        PatchReparam.decode internally handles denormalization and recon_proj (if configured).
        Output is [0, 1]; we convert to [-1, 1] to match Flux VAE.
        """
        if self._use_varlen_encode:
            B, C, h, w = z.shape
            grid_thw = torch.tensor(
                [[1, h, w]] * B, dtype=torch.long, device=z.device
            )
            # decode_varlen expects [total_tokens, latent_dim]
            z_flat = z.reshape(B, C, h * w).permute(0, 2, 1).reshape(B * h * w, C)
            x_rec = self.unified_vae.decode_varlen(z_flat, grid_thw)
            if isinstance(x_rec, list):
                x_rec = torch.stack(x_rec, dim=0)
            return (x_rec * 2.0 - 1.0).to(z.dtype)  # [0, 1] → [-1, 1]

        from stage1.patch_reparam import EncodeOutput, DecodeOutput
        enc_output = EncodeOutput(z=z)
        result = self.unified_vae.decode(enc_output)
        if isinstance(result, DecodeOutput):
            x_rec = result.x_rec
        else:
            x_rec = result
        return x_rec * 2.0 - 1.0  # [0, 1] → [-1, 1]


def _resolve_relative_path(p, project_root):
    """Resolve a relative path against project_root; return as-is if absolute or not found."""
    import os
    if p is None or os.path.isabs(str(p)):
        return p
    resolved = os.path.join(project_root, str(p))
    if os.path.exists(resolved):
        return resolved
    return p


def _detect_patch_reparam_root(raw_config, config_dir):
    """Infer PatchReparam project root from config metadata or directory structure."""
    import os
    # Try cmd_args.config first (saved by PatchReparam training)
    cmd_config = (raw_config.get("cmd_args", {}).get("config", None)
                  if OmegaConf.is_config(raw_config) else None)
    if cmd_config and os.path.isabs(str(cmd_config)):
        candidate = os.path.dirname(os.path.dirname(os.path.dirname(str(cmd_config))))
        if os.path.isdir(candidate):
            return candidate
    # Walk up from config_dir to find the parent of the "configs" or "ckpts" directory.
    # e.g. .../patch-reparameterization/configs/stage2/training/ImageNet256/ → .../patch-reparameterization
    #      .../patch-reparameterization/ckpts/stage1/<exp>/ → .../patch-reparameterization
    parts = os.path.abspath(config_dir).split(os.sep)
    for marker in ("configs", "ckpts"):
        if marker in parts:
            idx = parts.index(marker)
            candidate = os.sep + os.path.join(*parts[1:idx])
            if os.path.isdir(candidate):
                return candidate
    return os.path.dirname(config_dir)


def _resolve_patch_reparam_params(patch_reparam_params, patch_reparam_root):
    """Resolve relative paths inside PatchReparam params dict."""
    if OmegaConf.is_config(patch_reparam_params):
        patch_reparam_params = OmegaConf.to_container(patch_reparam_params, resolve=True)
    for key in ["decoder_config_path", "encoder_config_path", "normalization_stat_path"]:
        if key in patch_reparam_params:
            patch_reparam_params[key] = _resolve_relative_path(patch_reparam_params[key], patch_reparam_root)
    if "decoder" in patch_reparam_params and isinstance(patch_reparam_params["decoder"], dict):
        if "config_path" in patch_reparam_params["decoder"]:
            patch_reparam_params["decoder"]["config_path"] = _resolve_relative_path(
                patch_reparam_params["decoder"]["config_path"], patch_reparam_root)
    return patch_reparam_params


def _load_checkpoint_into_model(unified_vae, checkpoint_path, use_ema):
    """Load checkpoint weights into PatchReparam model, handling DDP/compile prefixes."""
    logger.info(f"Loading PatchReparam checkpoint from {checkpoint_path} (use_ema={use_ema})")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
        if use_ema and "ema" in checkpoint:
            state_dict = checkpoint["ema"]
            logger.info("Using EMA weights")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            logger.info("Using model weights")
        else:
            raise KeyError("Checkpoint has neither 'model' nor 'ema' keys")

        cleaned = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in ["module._orig_mod.", "_orig_mod.", "module."]:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
                    break
            cleaned[clean_k] = v

        missing, unexpected = unified_vae.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning(f"PatchReparam missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            logger.warning(f"PatchReparam unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    else:
        unified_vae.load_state_dict(checkpoint, strict=True)


def load_unified_vae(
    local_path: str,
    unified_vae_config_path: Optional[str] = None,
    use_ema: bool = True,
    patch_reparam_root: Optional[str] = None,
    encoder_only: bool = False,
) -> tuple:
    """Load PatchReparam encoder+decoder and return (UnifiedVAEWrapper, UnifiedVAEConfig).

    Supports two modes:

    1. **Config-driven** (recommended): ``local_path`` is the path to a PatchReparam
       config yaml (e.g. ``DiTDH-XL_siglip-B-unified-stage3-patchtune.yaml``).
       The config contains ``stage_1.params`` (model params) and
       ``stage_1.checkpoint.path`` / ``stage_1.checkpoint.use_ema`` (weights).
       ``unified_vae_config_path`` is ignored in this mode.

    2. **Legacy**: ``local_path`` is a checkpoint ``.pt`` file or experiment
       directory.  ``unified_vae_config_path`` points to the config yaml (or is
       auto-detected from the directory structure).

    Returns:
        (UnifiedVAEWrapper, UnifiedVAEConfig) tuple.
    """
    import os

    # --- Determine mode: config yaml vs checkpoint path/dir ---
    if local_path.endswith(('.yaml', '.yml')):
        # Mode 1: config-driven
        config_path = local_path
        logger.info(f"Loading PatchReparam from config: {config_path}")
        raw_config = OmegaConf.load(config_path)
        config_dir = os.path.dirname(os.path.abspath(config_path))
        # An explicit source root must win over config provenance. Release
        # checkpoints may retain configs stored under an older training tree.
        patch_reparam_root = patch_reparam_root or _detect_patch_reparam_root(
            raw_config, config_dir
        )
        logger.info(f"PatchReparam project root: {patch_reparam_root}")

        stage_1 = raw_config.get("stage_1", raw_config)
        patch_reparam_params = stage_1.get("params", stage_1)
        patch_reparam_params = _resolve_patch_reparam_params(patch_reparam_params, patch_reparam_root)

        # Checkpoint from config
        ckpt_cfg = stage_1.get("checkpoint", {})
        checkpoint_path = str(ckpt_cfg.get("path", ""))
        if not checkpoint_path:
            raise ValueError(f"No stage_1.checkpoint.path in {config_path}")
        checkpoint_path = _resolve_relative_path(checkpoint_path, patch_reparam_root)
        use_ema = ckpt_cfg.get("use_ema", use_ema)
    else:
        # Mode 2: legacy — local_path is checkpoint or experiment dir
        if os.path.isdir(local_path):
            exp_dir = local_path
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                ckpt_files = sorted(
                    [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                    key=lambda x: x,
                )
                if not ckpt_files:
                    raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
                checkpoint_path = os.path.join(ckpt_dir, ckpt_files[-1])
                logger.info(f"Using latest checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoints/ dir in {exp_dir}")
            if unified_vae_config_path is None:
                candidate = os.path.join(exp_dir, "config.yaml")
                if os.path.exists(candidate):
                    unified_vae_config_path = candidate
        else:
            checkpoint_path = local_path
            if unified_vae_config_path is None:
                parent = os.path.dirname(os.path.dirname(checkpoint_path))
                candidate = os.path.join(parent, "config.yaml")
                if os.path.exists(candidate):
                    unified_vae_config_path = candidate

        if unified_vae_config_path is None:
            raise ValueError(
                "Cannot find PatchReparam config.yaml. Provide unified_vae_config_path explicitly "
                "or pass a config yaml as local_path."
            )

        logger.info(f"Loading PatchReparam config from {unified_vae_config_path}")
        raw_config = OmegaConf.load(unified_vae_config_path)
        config_dir = os.path.dirname(os.path.abspath(unified_vae_config_path))
        patch_reparam_root = patch_reparam_root or _detect_patch_reparam_root(
            raw_config, config_dir
        )
        logger.info(f"PatchReparam project root: {patch_reparam_root}")

        patch_reparam_params = raw_config.get("stage_1", {}).get("params", raw_config)
        patch_reparam_params = _resolve_patch_reparam_params(patch_reparam_params, patch_reparam_root)

    # 动态 import patch-reparameterization（保证使用最新版本，不复制代码）
    _ensure_patch_reparam_importable(patch_reparam_root)
    from stage1.patch_reparam import PatchReparam
    unified_vae = PatchReparam(config=patch_reparam_params)

    # Load weights
    _load_checkpoint_into_model(unified_vae, checkpoint_path, use_ema)

    # Drop decoder to save memory when only encoding is needed
    if encoder_only and hasattr(unified_vae, 'decoder'):
        del unified_vae.decoder
        unified_vae.decoder = None
        logger.info("PatchReparam decoder dropped (encoder_only=True)")

    # Build output config
    unified_vae_config = UnifiedVAEConfig(
        downsample=unified_vae.encoder_patch_size,
        z_channels=unified_vae.latent_dim,
    )
    logger.info(f"PatchReparam config: downsample={unified_vae_config.downsample}, z_channels={unified_vae_config.z_channels}")

    wrapper = UnifiedVAEWrapper(unified_vae)
    return wrapper, unified_vae_config
