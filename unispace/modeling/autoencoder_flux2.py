# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted from diffusers 0.36 AutoencoderKLFlux2, standalone implementation.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained Flux2 VAE implementation that loads official BFL weights
# (diffusion_pytorch_model.safetensors) without requiring diffusers >= 0.36.
# Key names match diffusers convention exactly.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_sft


# ---------------------------------------------------------------------------
# Config dataclass (mirrors AutoEncoderParams from autoencoder.py for Flux1)
# ---------------------------------------------------------------------------
@dataclass
class Flux2VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    latent_channels: int = 32
    norm_num_groups: int = 32
    downsample: int = 8          # 2^(num_down_blocks - 1) = 2^3 = 8
    z_channels: int = 32         # alias for latent_channels, compat with Flux1 config
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    patch_size: tuple = (2, 2)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResnetBlock2D(nn.Module):
    """Standard ResNet block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + shortcut."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Self-attention block matching diffusers Attention with _from_deprecated_attn_block=True."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        x = self.group_norm(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.to_out[0](x)

        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x + residual


class Downsample2D(nn.Module):
    """Spatial downsample with padding + stride-2 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample2D(nn.Module):
    """Spatial upsample with nearest interpolation + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks (key names: down_blocks.{i}.resnets.{j} / downsamplers.0)
# ---------------------------------------------------------------------------
class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 2,
                 add_downsample: bool = True, groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            self.resnets.append(ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, groups))
        self.downsamplers = nn.ModuleList()
        if add_downsample:
            self.downsamplers.append(Downsample2D(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for ds in self.downsamplers:
            x = ds(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3,
                 add_upsample: bool = True, groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            self.resnets.append(ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, groups))
        self.upsamplers = nn.ModuleList()
        if add_upsample:
            self.upsamplers.append(Upsample2D(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for us in self.upsamplers:
            x = us(x)
        return x


class MidBlock(nn.Module):
    """Mid block: resnet -> attention -> resnet."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, groups),
            ResnetBlock2D(channels, channels, groups),
        ])
        self.attentions = nn.ModuleList([AttnBlock(channels, groups)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ---------------------------------------------------------------------------
# Encoder & Decoder
# ---------------------------------------------------------------------------
class Flux2Encoder(nn.Module):
    def __init__(self, cfg: Flux2VAEConfig):
        super().__init__()
        ch = cfg.block_out_channels
        self.conv_in = nn.Conv2d(cfg.in_channels, ch[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_c = ch[0]
        for i, out_c in enumerate(ch):
            is_last = (i == len(ch) - 1)
            self.down_blocks.append(
                DownEncoderBlock2D(in_c, out_c, cfg.layers_per_block,
                                   add_downsample=not is_last, groups=cfg.norm_num_groups)
            )
            in_c = out_c

        self.mid_block = MidBlock(ch[-1], cfg.norm_num_groups)

        self.conv_norm_out = nn.GroupNorm(cfg.norm_num_groups, ch[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch[-1], 2 * cfg.latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class Flux2Decoder(nn.Module):
    def __init__(self, cfg: Flux2VAEConfig):
        super().__init__()
        ch = cfg.block_out_channels
        rev_ch = list(reversed(ch))

        self.conv_in = nn.Conv2d(cfg.latent_channels, ch[-1], 3, padding=1)
        self.mid_block = MidBlock(ch[-1], cfg.norm_num_groups)

        self.up_blocks = nn.ModuleList()
        in_c = rev_ch[0]
        for i, out_c in enumerate(rev_ch):
            is_last = (i == len(rev_ch) - 1)
            # Decoder has layers_per_block + 1 resnets per block
            self.up_blocks.append(
                UpDecoderBlock2D(in_c, out_c, cfg.layers_per_block + 1,
                                 add_upsample=not is_last, groups=cfg.norm_num_groups)
            )
            in_c = out_c

        self.conv_norm_out = nn.GroupNorm(cfg.norm_num_groups, ch[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch[0], cfg.out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid_block(z)
        for block in self.up_blocks:
            z = block(z)
        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        z = self.conv_out(z)
        return z


# ---------------------------------------------------------------------------
# DiagonalGaussian (sample or mode)
# ---------------------------------------------------------------------------
class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean


# ---------------------------------------------------------------------------
# Main model: AutoEncoderFlux2
# ---------------------------------------------------------------------------
class AutoEncoderFlux2(nn.Module):
    """
    Standalone Flux2 VAE (AutoencoderKLFlux2) implementation.

    Key differences from Flux1 VAE (autoencoder.py):
      - latent_channels = 32 (vs 16)
      - Has quant_conv / post_quant_conv layers
      - Uses BatchNorm-based normalization instead of scale_factor/shift_factor
      - Architecture follows diffusers standard (DownEncoderBlock2D / UpDecoderBlock2D)
    """

    def __init__(self, config: Flux2VAEConfig | None = None):
        super().__init__()
        if config is None:
            config = Flux2VAEConfig()
        self.config = config

        self.encoder = Flux2Encoder(config)
        self.decoder = Flux2Decoder(config)
        self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)
        self.reg = DiagonalGaussian(sample=False)  # default: mode (deterministic)

        # BatchNorm for latent normalization (patchified space)
        bn_channels = math.prod(config.patch_size) * config.latent_channels  # 2*2*32=128
        self.bn = nn.BatchNorm2d(
            bn_channels,
            eps=config.batch_norm_eps,
            momentum=config.batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to normalized latent (BN in patchified space, then un-patchify)."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        z = self.reg(h)  # (B, C=32, H, W)

        # Patchify → BN normalize → un-patchify to keep API compatible with Flux1
        B, C, H, W = z.shape
        pH, pW = self.config.patch_size  # (2, 2)
        # (B, C, H, W) → (B, C*pH*pW, H//pH, W//pW)
        z_p = z.reshape(B, C, H // pH, pH, W // pW, pW)
        z_p = z_p.permute(0, 1, 3, 5, 2, 4).reshape(B, C * pH * pW, H // pH, W // pW)
        # BN normalize
        bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(z_p.device, z_p.dtype)
        bn_std = torch.sqrt(
            self.bn.running_var.view(1, -1, 1, 1).to(z_p.device, z_p.dtype)
            + self.config.batch_norm_eps
        )
        z_p = (z_p - bn_mean) / bn_std
        # Un-patchify: (B, C*pH*pW, H', W') → (B, C, H, W)
        z = z_p.reshape(B, C, pH, pW, H // pH, W // pW)
        z = z.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode normalized latent to image (symmetric with encode).

        Pipeline: patchify → BN denormalize → unpatchify → post_quant_conv → decoder.
        This mirrors the encode path which does: encoder → patchify → BN normalize → unpatchify.
        """
        B, C, H, W = z.shape
        pH, pW = self.config.patch_size  # (2, 2)

        # Patchify: (B, C, H, W) → (B, C*pH*pW, H//pH, W//pW)
        z_p = z.reshape(B, C, H // pH, pH, W // pW, pW)
        z_p = z_p.permute(0, 1, 3, 5, 2, 4).reshape(B, C * pH * pW, H // pH, W // pW)

        # BN denormalize
        bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(z_p.device, z_p.dtype)
        bn_std = torch.sqrt(
            self.bn.running_var.view(1, -1, 1, 1).to(z_p.device, z_p.dtype)
            + self.config.batch_norm_eps
        )
        z_p = z_p * bn_std + bn_mean

        # Un-patchify: (B, C*pH*pW, H', W') → (B, C, H, W)
        z = z_p.reshape(B, C, pH, pW, H // pH, W // pW)
        z = z.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)

        z = self.post_quant_conv(z)
        return self.decoder(z)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply BN normalization to patchified latents (as in Flux2Pipeline)."""
        bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            + self.config.batch_norm_eps
        )
        return (latents - bn_mean) / bn_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse BN normalization for decoding."""
        bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            + self.config.batch_norm_eps
        )
        return latents * bn_std + bn_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------
def _print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        print(f"[Flux2-VAE] {len(missing)} missing keys:\n\t" + "\n\t".join(missing[:20]))
    if unexpected:
        print(f"[Flux2-VAE] {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected[:20]))


def load_flux2_ae(local_path: str) -> tuple[AutoEncoderFlux2, Flux2VAEConfig]:
    """
    Load Flux2 VAE from a safetensors checkpoint directory or file.

    Args:
        local_path: Path to the safetensors file, or directory containing
                    diffusion_pytorch_model.safetensors.

    Returns:
        (model, config) tuple.
    """
    import os
    if os.path.isdir(local_path):
        sf_path = os.path.join(local_path, "diffusion_pytorch_model.safetensors")
    else:
        sf_path = local_path

    config = Flux2VAEConfig()
    model = AutoEncoderFlux2(config)

    sd = load_sft(sf_path)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    _print_load_warning(missing, unexpected)

    return model, config
