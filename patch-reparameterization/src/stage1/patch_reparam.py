import torch
import torch.nn as nn
from .decoders import DECODER_REGISTRY
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional, Union, Dict, Any, List
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
from math import sqrt
import dataclasses
from dataclasses import dataclass
import logging
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


@dataclass
class EncodeOutput:
    """Container for encoder outputs. Extensible for future aux data (side_tokens, etc.)."""
    z: torch.Tensor
    side_tokens: Optional[Dict[str, torch.Tensor]] = None
    debug_info: Optional[Dict[str, Any]] = None
    hidden_states_before_postnorm: Optional[torch.Tensor] = None
    # 扩展时可增加字段，如 attn_weights, intermediate_features 等
    sementic_tokens: Optional[torch.Tensor] = None
    sementic_tokens_before_postnorm: Optional[torch.Tensor] = None
    recon_tokens: Optional[torch.Tensor] = None
    merged_tokens: Optional[torch.Tensor] = None
    merged_tokens_norm: Optional[torch.Tensor] = None
    merged_tokens_with_global: Optional[torch.Tensor] = None
    image_features_1d: Optional[torch.Tensor] = None
    cls_label: Optional[torch.Tensor] = None  # optional label from encoder (e.g. from sampling) for decoder cls head
    map_attn_weights: Optional[torch.Tensor] = None  # (B, N) per-token semantic importance from MAP head attention
    # VAE related info: {"kl_loss": Tensor, "mu": Tensor, "logvar": Tensor, ...}
    # None if VAE is not enabled or not in training mode
    vae_info: Optional[Dict[str, torch.Tensor]] = None
    # vae_dist dict for precompute/offline sampling (mode C).
    # Keys: {"mu": Tensor, "logvar": Tensor}, both in norm space.
    # Populated when the encoder has a VAE head, aux is requested, and
    # the encoder supports _build_vae_dist (e.g. SigLIP2wTwoBackboneVAE).
    vae_dist: Optional[Dict[str, torch.Tensor]] = None

    sementic_z: Optional[torch.Tensor] = None # vae merge前的sementic token时
    recon_z: Optional[torch.Tensor] = None# vae merge前的
    recon_z_weighted: Optional[torch.Tensor] = None# vae merge前的，已经weight过的recon_tokens
    # AdaLN condition vector from recon tokens (used by GeneralDecoder_adaln)
    recon_cond: Optional[torch.Tensor] = None
    # Raw tokens before TokenMerger (for analysis / regularization)
    sementic_tokens_raw: Optional[torch.Tensor] = None
    recon_tokens_raw: Optional[torch.Tensor] = None
    # Denoise augmentation: per-sample t values [B, 1, ...] or None
    denoise_t: Optional[torch.Tensor] = None
    # Per-sample loss weights from noise_t_sampler (e.g. {'l1_scale': [B], 'gan_scale': [B]})
    loss_weights: Optional[Dict[str, torch.Tensor]] = None
@dataclass
class DecodeOutput:
    x_rec: torch.Tensor
    cls_classifier_output: Optional[torch.Tensor] = None
    decompose_loss: Optional[torch.Tensor] = None
    mid_x_rec: Optional[torch.Tensor] = None  # deep supervision: mid-layer reconstruction

def _load_pretrained_encoder(
    encoder: nn.Module,
    pretrained_path: Optional[str],
    pretrained_checkpoint: Optional[str],
    use_ema: bool = True,
) -> None:
    """
    加载 encoder 预训练权重。支持两种方式：
    1. pretrained_path: 纯 encoder state_dict 路径
    2. pretrained_checkpoint: 完整 PatchReparam checkpoint 路径，根据 use_ema 选择 model 或 ema 中的 encoder
    """
    if not (pretrained_path or pretrained_checkpoint):
        return

    if pretrained_path and pretrained_checkpoint:
        raise ValueError(
            "Cannot specify both encoder_pretrained_path and encoder_pretrained_checkpoint. Choose one."
        )

    load_path = pretrained_checkpoint if pretrained_checkpoint else pretrained_path

    checkpoint = torch.load(load_path, map_location="cpu")

    if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
        # 完整 PatchReparam checkpoint：根据 use_ema 选择 model 或 ema
        if use_ema:
            if "ema" not in checkpoint:
                raise KeyError(
                    "Checkpoint has no 'ema' key. Set encoder_pretrained_use_ema=False or use a different checkpoint."
                )
            state_dict = checkpoint["ema"]
        else:
            if "model" not in checkpoint:
                raise KeyError(
                    "Checkpoint has no 'model' key. Set encoder_pretrained_use_ema=True or use a different checkpoint."
                )
            state_dict = checkpoint["model"]

        # 提取 encoder 部分权重（处理各种包装前缀）
        encoder_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in [
                "module._orig_mod.encoder.",
                "_orig_mod.encoder.",
                "module.encoder.",
                "encoder.",
            ]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break

            if clean_key != key or key.startswith("encoder."):
                encoder_state_dict[clean_key] = value

        if len(encoder_state_dict) == 0:
            raise ValueError(
                "No encoder weights found in checkpoint. Check key prefixes."
            )

        missing_keys, unexpected_keys = encoder.load_state_dict(
            encoder_state_dict, strict=False
        )
        if missing_keys:
            logger.warning(
                f"[PatchReparam] Encoder missing keys when loading from checkpoint: {missing_keys}"
            )
        if unexpected_keys:
            logger.warning(
                f"[PatchReparam] Encoder unexpected keys when loading from checkpoint: {unexpected_keys}"
            )
        logger.info(
            f"Loaded pretrained encoder from full PatchReparam checkpoint {load_path} (use_ema={use_ema})"
        )
    else:
        # 纯 encoder state_dict
        encoder.load_state_dict(checkpoint, strict=True)
        logger.info(f"Loaded pretrained encoder from {load_path}")


class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class PatchReparam(nn.Module):
    def __init__(self, config: Union[DictConfig, Dict, Any] = None, **kwargs):

        super().__init__()
        # 兼容性处理：如果 config 为 None，尝试使用 kwargs (旧调用方式)
        if config is None:
            config = kwargs

        self.config = config
        # 辅助函数：安全地获取配置值，支持 OmegaConf(属性访问) 和 Dict(键访问)
        def get_cfg(obj, key, default=None):
            if obj is None: return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        # ====================================================
        # 1. Encoder Setup
        # ====================================================
        # 优先查找 config.encoder 节点，如果没有则回退到 config.encoder_xxx (扁平结构)
        enc_cfg = get_cfg(config, 'encoder', {})

        # 获取 Encoder 参数
        encoder_cls_name = get_cfg(enc_cfg, 'cls') or get_cfg(config, 'encoder_cls', 'Dinov2withNorm')
        encoder_config_path = get_cfg(enc_cfg, 'config_path') or get_cfg(config, 'encoder_config_path', 'facebook/dinov2-base')
        encoder_input_size = get_cfg(enc_cfg, 'input_size') or get_cfg(config, 'encoder_input_size', 224)

        # 获取实例化参数 params (例如 LongCatCLIP 需要的 model_config)
        encoder_params = get_cfg(enc_cfg, 'params') or get_cfg(config, 'encoder_params', {})
        # 如果是 OmegaConf 对象，转换为普通 dict 以便解包 (**params)
        if OmegaConf.is_config(encoder_params):
            encoder_params = OmegaConf.to_container(encoder_params, resolve=True)

        # 初始化 Encoder
        if encoder_cls_name not in ARCHS:
            raise ValueError(f"Encoder class {encoder_cls_name} not found in ARCHS registry.")
        encoder_cls = ARCHS[encoder_cls_name]

        # 这里会将 params 字典解包传入 Encoder 的 __init__
        # 例如: LongCatCLIPEncoder(model_config=...)
        logger.debug(f"encoder_params: {encoder_params}")
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        logger.info(f"Encoder initialized: {encoder_cls_name}")
        if encoder_cls_name == "LongCatCLIPEncoder":
            # LongCatCLIP 的 image_mean 和 image_std 存储在 config 中
            self.encoder_mean = torch.tensor(self.encoder.model.config.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(self.encoder.model.config.image_std).view(1, 3, 1, 1)
        elif getattr(self.encoder, "handles_preprocessing", False):
            # Encoder 内部处理预处理，从 encoder 获取 mean/std/input_size
            self.encoder_mean = self.encoder.encoder_mean
            self.encoder_std = self.encoder.encoder_std
            encoder_input_size = int(self.encoder.encoder_input_size)
        else:
            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
            self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
            # 对于其他 encoder，加载 config（LongCatCLIPEncoder 不需要，因为 config 已在 encoder 中）
            encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = get_cfg(config, 'decoder_latent_dim') or getattr(self.encoder, "recon_hidden_size", self.encoder.hidden_size)
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent

        encoder_pretrained_path = get_cfg(enc_cfg, 'encoder_pretrained_path') or get_cfg(config, 'encoder_pretrained_path')
        encoder_pretrained_checkpoint = get_cfg(enc_cfg, 'encoder_pretrained_checkpoint') or get_cfg(config, 'encoder_pretrained_checkpoint')
        encoder_use_ema = get_cfg(enc_cfg, 'encoder_pretrained_use_ema')
        if encoder_use_ema is None:
            encoder_use_ema = get_cfg(config, 'encoder_pretrained_use_ema', True)
        _load_pretrained_encoder(
            self.encoder,
            encoder_pretrained_path,
            encoder_pretrained_checkpoint,
            encoder_use_ema,
        )

        logger.info(f"encoder_mean: {self.encoder_mean}")
        logger.info(f"encoder_std: {self.encoder_std}")
        logger.info(f"encoder_input_size: {self.encoder_input_size}")
        logger.info(f"encoder_patch_size: {self.encoder_patch_size}")
        logger.info(f"latent_dim: {self.latent_dim}")
        logger.info(f"base_patches: {self.base_patches}")
        # decoder

        # ====================================================
        # 2. Decoder Setup
        # ====================================================
        dec_cfg = get_cfg(config, 'decoder', {})

        decoder_config_path = get_cfg(dec_cfg, 'config_path') or get_cfg(config, 'decoder_config_path', 'vit_mae-base')
        decoder_patch_size = get_cfg(dec_cfg, 'patch_size') or get_cfg(config, 'decoder_patch_size', 16)
        decoder_cls_name = get_cfg(dec_cfg, 'cls') or get_cfg(config, 'decoder_cls', 'GeneralDecoder')

        # 加载基础 HF Config
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))

        # 【关键】将 config.decoder 中的所有其他参数注入到 decoder_config 中
        # 这样 YAML 里写的 use_side_tokens, side_tokens_num 等都能自动传进去
        if isinstance(dec_cfg, (dict, DictConfig)):
            dec_cfg_dict = OmegaConf.to_container(dec_cfg, resolve=True) if OmegaConf.is_config(dec_cfg) else dec_cfg
            for k, v in dec_cfg_dict.items():
                if k not in ['cls', 'config_path', 'patch_size']: # 跳过已处理的
                    setattr(decoder_config, k, v)

        # 处理预训练权重路径 (支持从 decoder 节点或 root 读取)
        pretrained_ckpt = get_cfg(dec_cfg, 'pretrained_checkpoint') or get_cfg(config, 'pretrained_decoder_checkpoint')
        pretrained_path = get_cfg(dec_cfg, 'pretrained_path') or get_cfg(config, 'pretrained_decoder_path')
        use_ema = get_cfg(dec_cfg, 'pretrained_use_ema')
        if use_ema is None: use_ema = get_cfg(config, 'pretrained_decoder_use_ema', True)

        # 将路径设置到 decoder_config 中，GeneralDecoder 会在初始化时自动加载
        if pretrained_ckpt:
            decoder_config.pretrained_decoder_checkpoint = pretrained_ckpt
            decoder_config.pretrained_decoder_use_ema = use_ema
        if pretrained_path:
            decoder_config.pretrained_decoder_path = pretrained_path

        # 实例化 Decoder (via registry)
        if decoder_cls_name not in DECODER_REGISTRY:
            raise ValueError(
                f"Decoder '{decoder_cls_name}' not found in DECODER_REGISTRY. "
                f"Available: {list(DECODER_REGISTRY.keys())}"
            )
        decoder_cls = DECODER_REGISTRY[decoder_cls_name]
        logger.info(f"Using {decoder_cls_name}")
        self.decoder = decoder_cls(decoder_config, num_patches=self.base_patches)

        # ====================================================
        # 3. PatchReparam Global Setup
        # ====================================================
        self.noise_tau = get_cfg(config, 'noise_tau', 0.8)
        self.noise_tau_min = get_cfg(config, 'noise_tau_min', 0.0)
        self.noise_mode = get_cfg(config, 'noise_mode', 'additive')  # 'additive', 'flow', or 'denoise'
        self.denoise_steps = get_cfg(config, 'denoise_steps', 1)  # euler steps for denoise mode
        self.denoise_dit_config = get_cfg(config, 'denoise_dit_config', None)  # DiT config path for denoise mode
        # t sampler config: overrides noise_tau/noise_tau_min if present
        self.noise_t_sampler_cfg = get_cfg(config, 'noise_t_sampler', None)
        if self.noise_t_sampler_cfg is not None:
            logger.info(f"noise_t_sampler: {dict(self.noise_t_sampler_cfg)}, noise_mode: {self.noise_mode}")
        else:
            logger.info(f"noise_tau: [{self.noise_tau_min}, {self.noise_tau}], noise_mode: {self.noise_mode}")
        self.reshape_to_2d = get_cfg(config, 'reshape_to_2d', True)
        self.decode_noise= get_cfg(config, 'decode_noise', False)
        norm_stat_path = get_cfg(config, 'normalization_stat_path')
        self.eps = get_cfg(config, 'eps', 1e-5)

        if norm_stat_path:
            stats = torch.load(norm_stat_path, map_location='cpu')
            mean_raw = stats.get('mean', None)
            var_raw  = stats.get('var',  None)

            # norm_mode controls how the [N,D] stats are reduced before use:
            #   "token"  (default) : keep [N,D] as-is  -> per-token-per-dim normalisation
            #   "dim"              : mean over N -> [D] -> per-dim normalisation (shared across positions)
            #   "global"           : mean over N and D -> scalar -> global normalisation
            norm_mode = get_cfg(config, 'norm_mode', 'token')
            if norm_mode == 'dim':
                if mean_raw is not None and mean_raw.dim() >= 2:
                    mean_raw = mean_raw.mean(dim=0)   # [N,D] -> [D]  (or [C,H,W] -> [H,W] -> handled below)
                if var_raw is not None and var_raw.dim() >= 2:
                    var_raw  = var_raw.mean(dim=0)
                logger.info(f"norm_mode=dim: reduced stats to shape {tuple(mean_raw.shape) if mean_raw is not None else None}")
            elif norm_mode == 'global':
                if mean_raw is not None:
                    mean_raw = mean_raw.mean()        # scalar
                if var_raw is not None:
                    var_raw  = var_raw.mean()         # scalar
                logger.info(f"norm_mode=global: reduced stats to scalar")
            else:
                # "token": keep original shape
                logger.info(f"norm_mode=token: keeping stats shape {tuple(mean_raw.shape) if mean_raw is not None else None}")

            self.latent_mean = mean_raw
            self.latent_var  = var_raw
            logger.debug(f"latent mean:{self.latent_mean}")
            logger.debug(f"latent var:{self.latent_var}")
            self.do_normalization = True
            logger.info(f"Loaded normalization stats from {norm_stat_path}")
            self.norm_feature = get_cfg(config, 'norm_feature', 'z')
        else:
            self.do_normalization = False
            self.latent_mean = None
            self.latent_var = None
        self.need_recon_proj = get_cfg(config, 'need_recon_proj', False)
        self.need_sementic_proj = get_cfg(config, 'need_sementic_proj', False)
        self.need_recon_postnorm = get_cfg(config, 'need_recon_postnorm', False)

        self.force_noising = get_cfg(config, 'force_noising', False)
        self.dec_remove_global_token = get_cfg(config, 'dec_remove_global_token', False)

        if hasattr(self.encoder, "use_global_token"):
            assert self.encoder.use_global_token or self.dec_remove_global_token == self.encoder.use_global_token, "if dec_remove_global_token, must be encoder.use_global_token"
        assert not (self.need_recon_proj and self.need_sementic_proj), "need_recon_proj and need_sementic_proj cannot be True at the same time"

    def sample_t(self, batch_size: int, device: torch.device):
        """Sample per-sample t values and optional per-sample loss weights.

        Returns:
            t: Tensor[batch_size]
            loss_weights: Optional[Dict[str, Tensor[batch_size]]] — keys like 'l1_scale', 'lpips_scale', 'gan_scale'
        """
        cfg = self.noise_t_sampler_cfg
        if cfg is None:
            # Legacy: U[noise_tau_min, noise_tau], no loss weights
            t = self.noise_tau_min + (self.noise_tau - self.noise_tau_min) * torch.rand(batch_size, device=device)
            return t, None
        sampler_type = cfg['type']
        if sampler_type == 'uniform':
            t = cfg['min'] + (cfg['max'] - cfg['min']) * torch.rand(batch_size, device=device)
            lw_cfg = cfg.get('loss_weights', None)
            if lw_cfg is None:
                return t, None
            # Linear interpolation: t=min → range[0], t=max → range[1]
            alpha = (t - cfg['min']) / (cfg['max'] - cfg['min'] + 1e-8)
            loss_weights = {}
            for k, v_range in lw_cfg.items():
                loss_weights[k] = v_range[0] + alpha * (v_range[1] - v_range[0])
            return t, loss_weights
        elif sampler_type == 'discrete':
            values = cfg['values']
            if isinstance(values[0], (int, float)):
                # Old format: plain list of floats, no loss weights
                t_vals = torch.tensor(values, device=device, dtype=torch.float32)
                idx = torch.randint(len(t_vals), (batch_size,), device=device)
                return t_vals[idx], None
            else:
                # New format: list of dicts with 't' and optional loss weight keys
                t_vals = torch.tensor([v['t'] for v in values], device=device, dtype=torch.float32)
                idx = torch.randint(len(values), (batch_size,), device=device)
                t = t_vals[idx]
                weight_keys = [k for k in values[0].keys() if k != 't']
                if not weight_keys:
                    return t, None
                loss_weights = {}
                for k in weight_keys:
                    w_vals = torch.tensor([v.get(k, 1.0) for v in values], device=device, dtype=torch.float32)
                    loss_weights[k] = w_vals[idx]
                return t, loss_weights
        elif sampler_type == 'fixed':
            t = torch.full((batch_size,), cfg['value'], device=device)
            lw_cfg = cfg.get('loss_weights', None)
            if lw_cfg is None:
                return t, None
            loss_weights = {k: torch.full((batch_size,), v, device=device) for k, v in lw_cfg.items()}
            return t, loss_weights
        else:
            raise ValueError(f"Unknown noise_t_sampler type: {sampler_type}")

    def get_recon_last_layer(self) -> Optional[torch.nn.Parameter]:
        """Return the last trainable parameter in the recon token pathway.

        Used for adaptive weight calculation (e.g. diffusion prior regularization).
        Delegates to encoder.get_recon_last_layer() if available, otherwise falls
        back to encoder.get_last_layer().
        """
        if hasattr(self.encoder, 'get_recon_last_layer'):
            return self.encoder.get_recon_last_layer()
        if hasattr(self.encoder, 'get_last_layer'):
            return self.encoder.get_last_layer()
        return None

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        t, _ = self.sample_t(x.size(0), x.device)
        # reshape to [B, 1, ...] for broadcast
        t = t.view((x.size(0),) + (1,) * (len(x.shape) - 1))
        noise = torch.randn_like(x)
        if self.noise_mode == 'flow':
            # flow matching interpolation: (1-t)*data + t*noise
            return (1 - t) * x + t * noise
        else:
            # additive: data + t*noise
            return x + t * noise

    def normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the global mean/var normalisation to a pre-computed latent tensor.

        This is the same operation performed inside ``encode`` when
        ``do_normalization=True`` and ``norm_feature='z'``.  Exposing it as a
        standalone method lets the DiT training loop normalise cached (precomputed)
        latents without going through the full encode pipeline.

        If normalization is not configured (``do_normalization=False``), the tensor
        is returned unchanged.
        """
        if not self.do_normalization:
            return z

        latent_mean = self.latent_mean.to(z.device, z.dtype) if self.latent_mean is not None else 0
        latent_var = self.latent_var.to(z.device, z.dtype) if self.latent_var is not None else 1
        return (z - latent_mean) / torch.sqrt(latent_var + self.eps)

    def encode(self, x: torch.Tensor, return_aux: bool = False, debug: bool = False, image_features_mode: Optional[str] = None, force_sample: Optional[bool] = None, force_merge_train: bool = False) -> Union[torch.Tensor, EncodeOutput]:
        """
        Encode images to latent z. When return_aux=True, returns EncodeOutput(z, side_tokens)
        for use with decoders that consume side tokens (e.g. from learnable queries).
        When return_aux=False, returns only z (unchanged behavior).
        image_features_mode: for SigLIP2 encoder, override which tokens to use ("recon" | "merged" | "sementic").
        force_sample: if not None, override VAE sampling mode (True=sample, False=mu).
        force_merge_train: if True, force merge_tokens to use training-mode sampling (bool/bool_noise) even in eval mode.
        """
        # Qwen3 encoder: route through encode_varlen
        if self._is_qwen3_encoder():
            pixel_values, grid_thw = self._images_to_varlen(x)
            z_norm = self.encode_varlen(pixel_values, grid_thw)  # [B*N, C]
            # reshape_to_2d: [B*N, C] -> [B, N, C] -> [B, C, H, W]
            if self.reshape_to_2d:
                B = x.shape[0]
                N = z_norm.shape[0] // B
                h = w = int(N ** 0.5)
                c = z_norm.shape[-1]
                z_norm = z_norm.view(B, N, c).transpose(1, 2).view(B, c, h, w)
            if return_aux:
                return EncodeOutput(z=z_norm)
            return z_norm

        if not getattr(self.encoder, "handles_preprocessing", False):
            _, _, h, w = x.shape
            if h != self.encoder_input_size or w != self.encoder_input_size:
                x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
            x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

        # Call encoder with return_aux flag and optional image_features_mode (eval override)
        out = self.encoder(images=x, return_aux=return_aux, debug=debug, image_features_mode=image_features_mode, force_sample=force_sample, force_merge_train=force_merge_train)
        if isinstance(out, tuple) and len(out) == 2:
            z, aux = out
        else:
            z = out
            aux = {}

        image_features_1d = z

        denoise_t_value = None
        sampled_loss_weights = None
        if (self.training or self.force_noising) and (self.noise_tau > 0 or self.noise_t_sampler_cfg is not None):
            logger.debug(f"noising z: {z.shape}")
            if self.noise_mode == 'denoise':
                # Denoise mode: per-sample t, flow interpolation
                # DiT denoise step happens in forward()
                t, sampled_loss_weights = self.sample_t(z.size(0), z.device)
                t = t.view((z.size(0),) + (1,) * (len(z.shape) - 1))
                denoise_t_value = t  # [B, 1, ...] tensor, per-sample t
                noise = torch.randn_like(z)
                z = (1 - t) * z + t * noise
            else:
                z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)

        merged_tokens_norm = None
        if self.do_normalization:
            if self.norm_feature == 'z':
                latent_mean = self.latent_mean.to(z.device, z.dtype) if self.latent_mean is not None else 0
                latent_var = self.latent_var.to(z.device, z.dtype) if self.latent_var is not None else 1

                if z.dim() == 4: #已经reshape过了
                    h, w = z.shape[-2:]
                    if isinstance(latent_mean, torch.Tensor) and latent_mean.dim() == 2:
                        latent_mean = latent_mean.reshape(h, w, latent_mean.shape[-1]).permute(2,0,1)
                    if isinstance(latent_var, torch.Tensor) and latent_var.dim() == 2:
                        latent_var = latent_var.reshape(h, w, latent_var.shape[-1]).permute(2,0,1)


                z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
                logger.debug("norm z")
            else:
                # 这里不局限于mergerd tokens，但是变量名先用这个
                merged_tokens = aux.get(self.norm_feature, None)

                if merged_tokens is not None:
                    latent_mean = self.latent_mean.to(z.device, z.dtype) if self.latent_mean is not None else 0
                    latent_var = self.latent_var.to(z.device, z.dtype) if self.latent_var is not None else 1
                    # 若 latent_mean 为 [C, H, W]，转为 [N, C] 以匹配 merged_tokens [B, N, C]
                    # if isinstance(latent_mean, torch.Tensor) and latent_mean.dim() == 3:
                    #     latent_mean = latent_mean.permute(1, 2, 0).reshape(-1, latent_mean.shape[0])
                    # if isinstance(latent_var, torch.Tensor) and latent_var.dim() == 3:
                    #     latent_var = latent_var.permute(1, 2, 0).reshape(-1, latent_var.shape[0])

                    merged_tokens_norm = (merged_tokens - latent_mean) / torch.sqrt(latent_var + self.eps)
                    if self.reshape_to_2d:
                        b, n, c = merged_tokens_norm.shape
                        h = w = int(sqrt(n))
                        merged_tokens_norm = merged_tokens_norm.transpose(1, 2).view(b, c, h, w)
        # Extract VAE info if present (kl_loss, wasserstein_loss, mu, logvar, etc.)
        vae_info = None
        if aux.get("kl_loss") is not None or aux.get("wasserstein_loss") is not None:
            vae_info = {
                "kl_loss": aux.get("kl_loss"),
                "wasserstein_loss": aux.get("wasserstein_loss"),
                "mu": aux.get("mu"),
                "logvar": aux.get("logvar"),
                "recon_z": aux.get("recon_z"),
            }

        # Propagate vae_dist dict from encoder aux (keys: mu, logvar; both norm space).
        # The encoder is responsible for building this dict via _build_vae_dist.
        vae_dist = aux.get("vae_dist", None)  # plain dict or None

        if not return_aux:
            return z
        return EncodeOutput(z=z, side_tokens=aux.get("side_tokens"),
            debug_info=aux.get("debug_info"),
            hidden_states_before_postnorm=aux.get("hidden_states_before_postnorm", None),
            sementic_tokens=aux.get("sementic_tokens", None),

            sementic_z=aux.get("sementic_z", None),
            recon_z = aux.get("recon_z", None),

            recon_z_weighted=aux.get("recon_z_weighted", None), #这里一定要用weight后的

            sementic_tokens_before_postnorm=aux.get("sementic_tokens", None),
            recon_tokens=aux.get("recon_tokens", None),
            merged_tokens_with_global=aux.get("merged_tokens_with_global", None),
            image_features_1d=image_features_1d,
            merged_tokens=aux.get("merged_tokens", None),
            merged_tokens_norm=merged_tokens_norm,
            cls_label=aux.get("cls_label", None),
            map_attn_weights=aux.get("map_attn_weights", None),
            vae_info=vae_info,
            vae_dist=vae_dist,
            recon_cond=aux.get("recon_cond", None),
            sementic_tokens_raw=aux.get("sementic_tokens_raw", None),
            recon_tokens_raw=aux.get("recon_tokens_raw", None),
            denoise_t=denoise_t_value,
            loss_weights=sampled_loss_weights)

    def decode(
        self,
        encoder_output: Union[torch.Tensor, EncodeOutput],
        debug: bool = False,
        return_decoder_output: bool = False,
        decode_noise: bool = None,
    ) -> Union[torch.Tensor, DecodeOutput]:
        """
        Decode latent z (and optional side_tokens) to reconstructed image.
        Accepts Union[Tensor, EncodeOutput]; when EncodeOutput, uses .z and .side_tokens.
        Explicit side_tokens overrides EncodeOutput.side_tokens when both provided.
        """
        if isinstance(encoder_output, EncodeOutput):
            st = encoder_output.side_tokens
            z = encoder_output.z
            debug_info = encoder_output.debug_info
            recon_cond = encoder_output.recon_cond
            sementic_z = encoder_output.sementic_z
            recon_z_weighted = encoder_output.recon_z_weighted
        else:
            st = None
            debug_info = None
            z = encoder_output
            recon_cond = None
            sementic_z = None
            recon_z_weighted = None
        logger.debug(f"z {z.shape}")


        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)

        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device, z.dtype) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device, z.dtype) if self.latent_var is not None else 1
            # 若 latent_mean 为 [C, H, W]，转为 [N, C] 以匹配 z [B, N, C]
            if isinstance(latent_mean, torch.Tensor) and latent_mean.dim() == 3:
                latent_mean = latent_mean.permute(1, 2, 0).reshape(-1, latent_mean.shape[0])
            if isinstance(latent_var, torch.Tensor) and latent_var.dim() == 3:
                latent_var = latent_var.permute(1, 2, 0).reshape(-1, latent_var.shape[0])

            logger.debug(f"z shape: {z.shape}")
            logger.debug(f"latent_mean shape: {latent_mean.shape if isinstance(latent_mean, torch.Tensor) else latent_mean}")
            logger.debug(f"latent_var shape: {latent_var.shape if isinstance(latent_var, torch.Tensor) else latent_var}")
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
            # print(z.max(), z.min())
        if hasattr(self.encoder, "use_global_token") and self.encoder.use_global_token and self.dec_remove_global_token:
            z = z[:, 1:]

        if self.need_recon_proj:
            logger.debug("recon proj")
            z = self.encoder.get_recon_tokens(z)
        elif self.need_sementic_proj:
            logger.debug("sementic proj")
            z = self.encoder.get_sementic_tokens_for_recon(z)
        elif self.need_recon_postnorm:
            logger.debug("recon postnorm")
            z = self.encoder.get_recon_postnorm(z)

        use_decode_noise = self.decode_noise if decode_noise is None else decode_noise
        if use_decode_noise:
            z = self.noising(z)

        # Build extra kwargs for decoder based on capabilities
        dec_kwargs = {}
        # Decoders that accept recon_cond (e.g. adaln, crossattn with external condition)
        if recon_cond is not None and getattr(self.decoder, "accepts_recon_cond", False):
            dec_kwargs["recon_cond"] = recon_cond
        # Decoders with decompose module that accept semantic_gt / recon_gt
        if getattr(self.decoder, "has_decompose", False):
            if sementic_z is not None:
                dec_kwargs["semantic_gt"] = sementic_z
            if recon_z_weighted is not None:
                dec_kwargs["recon_gt"] = recon_z_weighted

        # Decoders that may return decompose_loss / mid_logits need full DecodeOutput
        needs_full_output = getattr(self.decoder, "has_decompose", False) or getattr(self.decoder, "has_deep_supervision", False) or getattr(self.decoder, "has_pixel_residual", False)

        if debug:
            dec_out = self.decoder(z, drop_cls_token=False, side_tokens=st, debug=debug, output_attentions=True, **dec_kwargs)
            all_attention_scores = dec_out.attentions
            logits = dec_out.logits
            decoder_attention_scores = [{"attention":attn} for attn in all_attention_scores]
            debug_info["decoder_attention_scores"] = decoder_attention_scores
            dec_out_for_cls = dec_out if return_decoder_output else None
        else:
            if return_decoder_output or needs_full_output:
                dec_out = self.decoder(z, drop_cls_token=False, side_tokens=st, **dec_kwargs)
                logits = dec_out.logits
                dec_out_for_cls = dec_out
            else:
                logits = self.decoder(z, drop_cls_token=False, side_tokens=st, **dec_kwargs).logits
                dec_out_for_cls = None
        x_rec = self.decoder.unpatchify(logits)
        if not getattr(self.decoder, "skip_rescale", False):
            x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)

        # Apply pixel-space residual if decoder provides it (e.g. CNN refiner)
        pixel_residual = getattr(dec_out_for_cls, "pixel_residual", None) if dec_out_for_cls is not None else None
        if pixel_residual is not None:
            x_rec = x_rec + pixel_residual

        # Extract decompose_loss and mid_logits if available
        decompose_loss = getattr(dec_out_for_cls, "decompose_loss", None) if dec_out_for_cls is not None else None
        mid_logits = getattr(dec_out_for_cls, "mid_logits", None) if dec_out_for_cls is not None else None

        # Deep supervision: unpatchify mid_logits to mid_x_rec
        mid_x_rec = None
        if mid_logits is not None:
            mid_x_rec = self.decoder.unpatchify(mid_logits)
            mid_x_rec = mid_x_rec * self.encoder_std.to(mid_x_rec.device) + self.encoder_mean.to(mid_x_rec.device)

        if return_decoder_output and (dec_out_for_cls is not None ):
            logger.debug("return decode output")
            return DecodeOutput(
                x_rec=x_rec,
                cls_classifier_output=getattr(dec_out_for_cls, "cls_classifier_output", None),
                decompose_loss=decompose_loss,
                mid_x_rec=mid_x_rec,
            )
        if debug:
            return x_rec, debug_info
        if decompose_loss is not None or mid_x_rec is not None:
            return DecodeOutput(x_rec=x_rec, decompose_loss=decompose_loss, mid_x_rec=mid_x_rec)
        return x_rec

    def get_proj(self, proj_name: str) -> Optional[nn.Module]:
        """
        Get a projection module from the encoder by name.
        For example, 'sementic_proj' or 'recon_proj'.
        """
        if hasattr(self.encoder, "merger"):
            # Try exact match first
            if hasattr(self.encoder.merger, proj_name):
                return getattr(self.encoder.merger, proj_name)
            # Try with 'sementic' typo if 'semantic' was requested
            if proj_name == 'semantic_proj' and hasattr(self.encoder.merger, 'sementic_proj'):
                return getattr(self.encoder.merger, 'sementic_proj')
        return None

    # ------------------------------------------------------------------
    # Varlen interface for token-packed variable-resolution training
    # (Qwen3Unified encoder + ViTDecoder with flash varlen attention)
    # ------------------------------------------------------------------

    def encode_varlen(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                      return_deepstack: bool = False,
                      normalize: bool = True):
        """Encode a varlen batch of images.

        Args:
            pixel_values:    [total_patches * temporal_patch_size, C, patch_size, patch_size]
            grid_thw:        [num_imgs, 3]  [T, H_grid, W_grid] per image
            return_deepstack: if True and encoder.use_deepstack, also return deepstack features
            normalize:       if False, skip normalize_latents (caller normalizes per-use).
                             Useful when gen and und tokens are encoded together: skip here,
                             then normalize only the gen portion after splitting.

        Returns:
            z: [total_tokens, latent_dim]  (normalized when normalize=True, raw when False)
            deepstack_features: list[Tensor] each [total_tokens/4, llm_hidden]
                                only returned when return_deepstack=True
        """
        out = self.encoder(pixel_values=pixel_values, grid_thw=grid_thw, return_aux=True)
        if isinstance(out, tuple):
            z, aux = out[0], out[1] if len(out) > 1 else {}
        else:
            z, aux = out, {}

        deepstack_features = aux.get('deepstack_features', None) if isinstance(aux, dict) else None

        if (self.training or self.force_noising) and self.noise_tau > 0:
            z = self.noising(z)

        if normalize:
            z = self.normalize_latents(z)

        if return_deepstack:
            return z, deepstack_features
        return z

    def decode_varlen(self, z_norm: torch.Tensor, grid_thw: torch.Tensor) -> List[torch.Tensor]:
        """Decode varlen latents back to images.

        When all images share the same (H, W) (e.g. from BucketBatchSampler),
        uses efficient batch mode. Falls back to per-image loop for mixed resolutions.

        Args:
            z_norm:   [total_tokens, latent_dim]
            grid_thw: [num_imgs, 3]

        Returns:
            list of [3, H_i, W_i] tensors in [0, 1]
        """
        p = self.decoder.config.patch_size
        B = grid_thw.shape[0]

        # denormalize latents (逆 normalize_latents)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z_norm.device, z_norm.dtype) if self.latent_mean is not None else 0
            latent_var  = self.latent_var.to(z_norm.device, z_norm.dtype)  if self.latent_var  is not None else 1
            z_norm = z_norm * torch.sqrt(latent_var + self.eps) + latent_mean

        # fast path: all same resolution → batch mode, returns [B, 3, H, W]
        unique_hw = set((int(grid_thw[i, 1]), int(grid_thw[i, 2])) for i in range(B))
        if len(unique_hw) == 1:
            h, w    = unique_hw.pop()
            n       = h * w
            z_batch = z_norm.reshape(B, n, -1)
            dec_out = self.decoder(z_batch, hw=(h, w))
            logits  = dec_out.logits
            x_recs  = self.decoder.unpatchify(logits, original_image_size=(h * p, w * p))
            x_recs  = x_recs * self.encoder_std.to(x_recs.device) + self.encoder_mean.to(x_recs.device)
            return x_recs  # [B, 3, H, W]

        # slow path: mixed resolutions → per-image loop
        dec_out     = self.decoder(z_norm, grid_thw=grid_thw)
        flat_logits = dec_out.logits
        images = []
        offset = 0
        for i in range(B):
            h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
            n        = h * w
            logits_i = flat_logits[offset: offset + n]
            x_rec    = self.decoder.unpatchify(logits_i, original_image_size=(h * p, w * p))
            x_rec    = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
            images.append(x_rec.squeeze(0))
            offset  += n
        return images

    def forward_varlen(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Varlen forward: encode → decode → list of reconstructed images.

        Args:
            pixel_values: [total_patches * tp, C, p, p]
            grid_thw:     [num_imgs, 3]

        Returns:
            list of [3, H_i, W_i] in [0, 1]
        """
        z_norm = self.encode_varlen(pixel_values, grid_thw)
        return self.decode_varlen(z_norm, grid_thw)

    def _is_qwen3_encoder(self) -> bool:
        from .encoders.qwen3_unified import Qwen3Unified
        from .encoders.qwen3_frozen import Qwen3Frozen
        enc = self.encoder
        # unwrap DDP/FSDP if needed
        if hasattr(enc, 'module'):
            enc = enc.module
        return isinstance(enc, (Qwen3Unified, Qwen3Frozen))

    def _images_to_varlen(self, x: torch.Tensor):
        """Convert [B, 3, H, W] images in [0,1] to Qwen3 pixel_values + grid_thw varlen format.
        Applies encoder normalization (mean=0.5, std=0.5) before patch extraction."""
        B, C, H, W = x.shape
        p  = self.encoder.patch_size
        tp = self.encoder.temporal_patch_size
        H_grid = H // p
        W_grid = W // p
        # normalize [0,1] -> [-1,1] using encoder mean/std
        mean = self.encoder_mean.to(x.device, dtype=x.dtype)
        std  = self.encoder_std.to(x.device, dtype=x.dtype)
        x = (x - mean) / std
        frames = x.unsqueeze(1).expand(-1, tp, -1, -1, -1)  # [B, tp, 3, H, W]
        frames = frames.reshape(B, tp, C, H_grid, p, W_grid, p)
        pixel_format = getattr(self.encoder, 'pixel_format', 't_major')
        if pixel_format == 'c_major':
            # C-major: [B, H_g, W_g, C, tp, p, p] → [B*N, C*tp*p*p]
            # 与官方 Qwen3VL processor 一致，PatchEmbed 内 view(-1, C, T, p, p) 能正确解读
            frames = frames.permute(0, 3, 5, 2, 1, 4, 6).reshape(B * H_grid * W_grid, C * tp * p * p)
        else:
            # T-major（旧行为，兼容已有 checkpoint）
            frames = frames.permute(0, 3, 5, 1, 2, 4, 6).reshape(B * H_grid * W_grid * tp, C, p, p)
        grid_thw = torch.tensor([[1, H_grid, W_grid]], dtype=torch.long, device=x.device).expand(B, -1)
        return frames, grid_thw

    def forward(self, x: torch.Tensor = None, debug: bool = False, return_encoder_output: bool = False, only_encoder_output: bool = False, image_features_mode: Optional[str] = None, return_decoder_output: bool = False, force_merge_train: bool = False,
                denoise_dit_model=None, denoise_labels=None, denoise_steps: int = 1,
                varlen: bool = False, pixel_values: torch.Tensor = None, grid_thw: torch.Tensor = None) -> torch.Tensor:
        if varlen:
            assert pixel_values is not None and grid_thw is not None, \
                "varlen=True requires pixel_values and grid_thw"
            return self.forward_varlen(pixel_values, grid_thw)

        # Auto-route: Qwen3 encoder can't accept [B,3,H,W] directly
        if x is not None and self._is_qwen3_encoder():
            pixel_values, grid_thw = self._images_to_varlen(x)
            if return_encoder_output or only_encoder_output:
                # Need encoder output — use encode_varlen then decode manually
                z_norm = self.encode_varlen(pixel_values, grid_thw)
                enc_out = EncodeOutput(z=z_norm)
                if only_encoder_output:
                    return enc_out
                x_rec = self.decode_varlen(z_norm, grid_thw)
                if isinstance(x_rec, list):
                    import torch as _t
                    x_rec = _t.stack(x_rec)
                if return_decoder_output:
                    from .decoders import DecodeOutput as _DecOut
                    dec_out = _DecOut(x_rec=x_rec)
                    return (x_rec, enc_out), dec_out
                return x_rec, enc_out
            return self.forward_varlen(pixel_values, grid_thw)

        enc = self.encode(x, return_aux=True, debug=debug, image_features_mode=image_features_mode, force_merge_train=force_merge_train)
        if only_encoder_output:
            return enc
        # Denoise augmentation: DiT denoise between encode and decode
        if denoise_dit_model is not None and enc.denoise_t is not None:
            logger.debug("denoise dit model")
            z = enc.z  # [B, C, H, W]
            # reshape t to match z: [B] -> [B, 1, 1, 1]
            t_per_sample = enc.denoise_t.view(-1)  # [B]
            t_broad = t_per_sample.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcast with z
            labels = denoise_labels.to(z.device) if denoise_labels is not None else None
            # Per-sample euler ODE: t_per_sample -> 0 (bf16 for DiT)
            orig_dtype = z.dtype
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                z_bf16 = z.to(torch.bfloat16)
                t_broad_bf16 = t_broad.to(torch.bfloat16)
                for step_i in range(denoise_steps):
                    t_cur_broad = t_broad_bf16 * (denoise_steps - step_i) / denoise_steps
                    dt = -t_broad_bf16 / denoise_steps  # [B, 1, 1, 1]
                    t_batch = t_cur_broad.view(-1)  # [B] for DiT t_embedder
                    v = denoise_dit_model(z_bf16, t_batch, y=labels)
                    z_bf16 = z_bf16 + v * dt
                z = z_bf16.to(orig_dtype)
            enc = dataclasses.replace(enc, z=z)
        res = self.decode(enc, debug=debug, return_decoder_output=return_decoder_output)
        if return_encoder_output:
            return res, enc
        return res