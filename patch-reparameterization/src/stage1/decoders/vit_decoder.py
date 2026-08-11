"""
Clean ViT decoder with:
  - On-the-fly 2D sin/cos positional embeddings (no interpolation)
  - Flash attention varlen for token-packed varlen batches
  - Any-resolution decode via grid_thw
"""

import math
import logging
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from .utils import ViTMAEConfig
from .decoder import ViTMAEDecoderOutput, ViTMAEIntermediate, ViTMAEOutput
from . import register_decoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Varlen attention: flash_attn when available, SDPA fallback otherwise.
# Copied from unimm-transfusion/modeling/unimm/qwen3.py to avoid import chain.
# ---------------------------------------------------------------------------
import os as _os
import torch.nn.functional as _F

def _get_attention_backend():
    env = _os.environ.get('ATTENTION_BACKEND', '').strip().lower()
    if env in ('flash_attn', 'torch_sdpa'):
        return env
    try:
        import flash_attn  # noqa
        return 'flash_attn'
    except ImportError:
        return 'torch_sdpa'

_ATTENTION_BACKEND = _get_attention_backend()
if _ATTENTION_BACKEND == 'flash_attn':
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func


def universal_varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    if _ATTENTION_BACKEND == 'flash_attn':
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        out = _flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            causal=causal, dropout_p=dropout_p,
        )
        return out.to(orig_dtype)
    # SDPA fallback (GPU without flash_attn, or NPU)
    q_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    k_seqlens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    q_list = q.split(q_seqlens, dim=0)
    k_list = k.split(k_seqlens, dim=0)
    v_list = v.split(k_seqlens, dim=0)
    outputs = []
    for qi, ki, vi in zip(q_list, k_list, v_list):
        qi = qi.permute(1, 0, 2).unsqueeze(0)
        ki = ki.permute(1, 0, 2).unsqueeze(0)
        vi = vi.permute(1, 0, 2).unsqueeze(0)
        ao = _F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None,
                                              dropout_p=dropout_p, is_causal=causal)
        outputs.append(ao.squeeze(0).permute(1, 0, 2))
    return torch.cat(outputs, dim=0)


def _build_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """[1, h*w, embed_dim] on-the-fly 2D sin/cos pos emb."""
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, -1)  # [2, h*w]

    assert embed_dim % 4 == 0
    half = embed_dim // 2

    def _1d(pos, dim):
        omega = np.arange(dim // 2, dtype=np.float64)
        omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
        out = np.outer(pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)

    emb = np.concatenate([_1d(grid[1], half), _1d(grid[0], half)], axis=1)  # [h*w, D]
    return torch.from_numpy(emb).unsqueeze(0).to(device=device, dtype=dtype)


class VarlenAttention(nn.Module):
    """Multi-head self-attention using universal_varlen_attention (flash_attn or SDPA fallback).

    Input: flat [total_tokens, hidden_size] + cu_seqlens.
    Compatible with both GPU (flash_attn) and NPU (SDPA fallback) via unimm's universal_varlen_attention.
    """

    def __init__(self, config: ViTMAEConfig):
        super().__init__()
        self.num_heads   = config.num_attention_heads
        self.head_dim    = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout_p   = config.attention_probs_dropout_prob

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key   = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        N = x.shape[0]
        q = self.query(x).reshape(N, self.num_heads, self.head_dim)
        k = self.key(x).reshape(N, self.num_heads, self.head_dim)
        v = self.value(x).reshape(N, self.num_heads, self.head_dim)

        out = universal_varlen_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # [total_tokens, num_heads, head_dim]

        return out.reshape(N, self.hidden_size)


class VarlenDecoderLayer(nn.Module):
    """ViT decoder layer using VarlenAttention + same FFN as ViTMAELayer."""

    def __init__(self, config: ViTMAEConfig):
        super().__init__()
        self.attn        = VarlenAttention(config)
        self.attn_out    = nn.Linear(config.hidden_size, config.hidden_size)
        self.intermediate = ViTMAEIntermediate(config)
        self.output       = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after  = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        # pre-norm attention
        attn_out = self.attn(self.layernorm_before(x), cu_seqlens, max_seqlen)
        attn_out = self.attn_out(attn_out)
        x = x + attn_out
        # FFN
        x = x + self.output(self.intermediate(self.layernorm_after(x)), x)
        return x


@register_decoder()
class ViTDecoder(nn.Module):
    """Clean ViT decoder with flash attention varlen.

    Calling modes:
      varlen:  x [total_tokens, hidden_size], grid_thw [num_imgs, 3]
      batch:   x [B, N, hidden_size], grid_thw=None (hw inferred or passed via hw=)
    """

    def __init__(self, config: ViTMAEConfig, num_patches: int):
        super().__init__()
        self.config      = config
        self.num_patches = num_patches
        self.use_sem_ln  = getattr(config, 'use_sem_ln', False)
        self.sem_dim     = getattr(config, 'sem_dim', None)
        if self.use_sem_ln and self.sem_dim is None:
            raise ValueError("ViTDecoder: use_sem_ln=True requires sem_dim to be set in decoder config")

        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, config.decoder_hidden_size))

        dec_cfg = deepcopy(config)
        dec_cfg.hidden_size        = config.decoder_hidden_size
        dec_cfg.num_hidden_layers  = config.decoder_num_hidden_layers
        dec_cfg.num_attention_heads= config.decoder_num_attention_heads
        dec_cfg.intermediate_size  = config.decoder_intermediate_size

        self.decoder_layers = nn.ModuleList(
            [VarlenDecoderLayer(dec_cfg) for _ in range(config.decoder_num_hidden_layers)]
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size ** 2 * config.num_channels,
            bias=True,
        )

        self.gradient_checkpointing = False
        self._init_weights()
        self._load_pretrained_weights(config)

        logger.info(
            f"ViTDecoder(flash): hidden={config.hidden_size}->{config.decoder_hidden_size}, "
            f"layers={config.decoder_num_hidden_layers}, patch_size={config.patch_size}"
        )

    def _init_weights(self):
        nn.init.constant_(self.decoder_pred.weight, 0)
        nn.init.constant_(self.decoder_pred.bias,   0)
        nn.init.normal_(self.trainable_cls_token, std=0.02)

    def _load_pretrained_weights(self, config):
        pretrained_path       = getattr(config, "pretrained_decoder_path", None)
        pretrained_checkpoint = getattr(config, "pretrained_decoder_checkpoint", None)
        use_ema               = getattr(config, "pretrained_decoder_use_ema", True)

        if not (pretrained_path or pretrained_checkpoint):
            return
        if pretrained_path and pretrained_checkpoint:
            raise ValueError("Cannot specify both pretrained_decoder_path and pretrained_decoder_checkpoint.")

        load_path = pretrained_checkpoint if pretrained_checkpoint else pretrained_path
        try:
            checkpoint = torch.load(load_path, map_location="cpu")
            if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
                state_dict = checkpoint["ema"] if use_ema else checkpoint["model"]
                decoder_state_dict = {}
                for key, value in state_dict.items():
                    clean_key = key
                    for prefix in ["module._orig_mod.decoder.", "_orig_mod.decoder.", "module.decoder.", "decoder."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    if clean_key != key or key.startswith("decoder."):
                        decoder_state_dict[clean_key] = value
                if not decoder_state_dict:
                    raise ValueError("No decoder weights found in checkpoint.")
            else:
                decoder_state_dict = checkpoint

            current_state = self.state_dict()
            filtered, skipped = {}, []
            for k, v in decoder_state_dict.items():
                if k in current_state and v.shape != current_state[k].shape:
                    skipped.append(f"{k}: ckpt {v.shape} vs model {current_state[k].shape}")
                else:
                    filtered[k] = v
            if skipped:
                logger.warning(f"[ViTDecoder] Skipping shape-mismatched keys: {skipped}")

            missing, unexpected = self.load_state_dict(filtered, strict=False)
            if missing:
                logger.warning(f"[ViTDecoder] Missing keys: {missing}")
            if unexpected:
                logger.warning(f"[ViTDecoder] Unexpected keys: {unexpected}")
            logger.info(f"[ViTDecoder] Loaded pretrained weights from {load_path}")
        except Exception as e:
            raise RuntimeError(f"[ViTDecoder] Failed to load pretrained weights from {load_path}: {e}")

    def _pos_embed(self, h: int, w: int) -> torch.Tensor:
        ref = self.decoder_pred.weight
        return _build_2d_sincos_pos_embed(self.config.decoder_hidden_size, h, w, ref.device, ref.dtype)
        # shape: [1, h*w, D]

    def _run_layers(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                def _ckpt(m, _x, _cu, _ms):
                    return m(_x, _cu, _ms)
                x = torch.utils.checkpoint.checkpoint(
                    _ckpt, layer, x, cu_seqlens, max_seqlen, use_reentrant=False
                )
            else:
                x = layer(x, cu_seqlens, max_seqlen)
        return x

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        drop_cls_token: bool = False,
        hw: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> ViTMAEDecoderOutput:
        D = self.config.decoder_hidden_size

        if self.use_sem_ln:
            sem = _F.layer_norm(x[..., :self.sem_dim], [self.sem_dim])
            x   = torch.cat([sem, x[..., self.sem_dim:]], dim=-1)

        if grid_thw is not None:
            # ── varlen mode ──────────────────────────────────────────────────
            # x: [total_tokens, hidden_size]
            x = self.decoder_embed(x)  # [total_tokens, D]

            # build per-image pos emb and prepend CLS, then concat all
            segments: List[torch.Tensor] = []
            seqlens:  List[int]           = []

            offset = 0
            for i in range(grid_thw.shape[0]):
                t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
                n = t * h * w
                xi    = x[offset: offset + n]             # [n, D]
                pos_i = self._pos_embed(h, w).squeeze(0)  # [n, D]
                cls_i = self.trainable_cls_token           # [D] → reshape to [1, D]
                cls_tok = cls_i.view(1, D)
                cls_pos = torch.zeros(1, D, device=x.device, dtype=x.dtype)
                seg = torch.cat([cls_tok + cls_pos, xi + pos_i], dim=0)  # [n+1, D]
                segments.append(seg)
                seqlens.append(n + 1)
                offset += n

            flat_x = torch.cat(segments, dim=0)  # [total_tokens + num_imgs, D]
            cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=x.device)
            cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=x.device).cumsum(0)
            max_seqlen = max(seqlens)

            flat_x = self._run_layers(flat_x, cu_seqlens, max_seqlen)
            flat_x = self.decoder_norm(flat_x)
            flat_x = self.decoder_pred(flat_x)  # [total+num_imgs, patch_size**2*C]

            # strip CLS from each image and return flat
            logits_list = []
            ptr = 0
            for sl in seqlens:
                logits_list.append(flat_x[ptr + 1: ptr + sl])  # skip CLS
                ptr += sl
            flat_logits = torch.cat(logits_list, dim=0)  # [total_tokens, patch_size**2*C]
            return ViTMAEDecoderOutput(logits=flat_logits)

        else:
            # ── batch mode (uniform resolution) ─────────────────────────────
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, N, _ = x.shape
            if hw is not None:
                h, w = hw
            else:
                h = w = int(math.sqrt(N))
                assert h * w == N

            x = self.decoder_embed(x)         # [B, N, D]
            pos = self._pos_embed(h, w)        # [1, N, D]
            cls = self.trainable_cls_token.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]  (wrong shape)

            # cls_token is [1, D] → reshape
            cls_tok = self.trainable_cls_token.view(1, 1, D).expand(B, 1, D)
            cls_pos = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            hidden  = torch.cat([cls_tok + cls_pos, x + pos], dim=1)  # [B, N+1, D]

            # build cu_seqlens for uniform batch
            seqlen = N + 1
            cu_seqlens = torch.arange(0, (B + 1) * seqlen, seqlen, dtype=torch.int32, device=x.device)
            hidden_flat = hidden.reshape(B * seqlen, D)

            hidden_flat = self._run_layers(hidden_flat, cu_seqlens, seqlen)
            hidden_flat = self.decoder_norm(hidden_flat)
            logits_flat = self.decoder_pred(hidden_flat)          # [B*(N+1), p**2*C]
            logits      = logits_flat.reshape(B, seqlen, -1)[:, 1:, :]  # [B, N, p**2*C]
            return ViTMAEDecoderOutput(logits=logits)

    def unpatchify(self, logits: torch.Tensor, original_image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        logits: [N, p**2*C]  or  [B, N, p**2*C]
        Returns [C, H, W] or [B, C, H, W]
        """
        p = self.config.patch_size
        c = self.config.num_channels
        squeeze = False
        if logits.dim() == 2:
            logits  = logits.unsqueeze(0)
            squeeze = True

        n = logits.shape[1]
        if original_image_size is not None:
            h_img, w_img = original_image_size
            nh, nw = h_img // p, w_img // p
        else:
            nh = nw = int(math.sqrt(n))
            assert nh * nw == n

        x = logits.reshape(logits.shape[0], nh, nw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(logits.shape[0], c, nh * p, nw * p)
        return x.squeeze(0) if squeeze else x
