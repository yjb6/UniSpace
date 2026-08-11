import copy
import logging

import torch
from torch import nn
from transformers import Dinov2WithRegistersModel

from . import register_encoder

logger = logging.getLogger(__name__)


@register_encoder()
class DINOv2Unified(nn.Module):
    """Single-backbone DINOv2-with-registers encoder with two independent embeddings.

    Architecture:
        image -> sem_embeddings   -> [B, 1+R+N, C] --|
                                                       |--> cat [2B, ...] -> encoder -> split
        image -> recon_embeddings -> [B, 1+R+N, C] --|

        sem_out   (patch tokens only) -> sem_ln                 -> [B, N, C]
        recon_out (patch tokens only) -> recon_proj -> recon_ln -> [B, N, recon_proj_dim]

        output: cat([sem, recon], dim=-1)  [B, N, C + recon_proj_dim]

    R = num_register_tokens (default 4), N = num_patches (default 256).
    CLS and register tokens are discarded after the encoder.
    """

    def __init__(self, model_name: str, recon_proj_dim: int = 256, **kwargs):
        super().__init__()

        backbone = Dinov2WithRegistersModel.from_pretrained(model_name)
        self.hidden_size = backbone.config.hidden_size
        self.patch_size = backbone.config.patch_size
        self.num_register_tokens = backbone.config.num_register_tokens
        self.recon_proj_dim = recon_proj_dim

        # Two independent embeddings (patch_embeddings + cls_token + register_tokens + position_embeddings)
        self.sem_embeddings   = copy.deepcopy(backbone.embeddings)
        self.recon_embeddings = copy.deepcopy(backbone.embeddings)
        # mask_token is only used in MAE pre-training; remove to avoid DDP unused-parameter error
        del self.recon_embeddings.mask_token
        self.recon_embeddings.mask_token = None

        # Shared encoder + layernorm
        self.encoder   = backbone.encoder
        self.layernorm = backbone.layernorm

        # recon projection + LN
        self.recon_proj = nn.Linear(self.hidden_size, recon_proj_dim, bias=True)
        nn.init.xavier_uniform_(self.recon_proj.weight)
        nn.init.zeros_(self.recon_proj.bias)

        self.sem_ln   = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.recon_ln = nn.LayerNorm(recon_proj_dim,   elementwise_affine=False)

        self.output_dim = self.hidden_size + recon_proj_dim
        self.recon_hidden_size = self.output_dim

        # skip tokens: 1 CLS + num_register_tokens
        self._skip = 1 + self.num_register_tokens

        logger.info(
            f"DINOv2Unified: hidden_size={self.hidden_size}, "
            f"recon_proj_dim={recon_proj_dim}, output_dim={self.output_dim}, "
            f"skip_tokens={self._skip}"
        )

    def forward(self, images: torch.Tensor, return_aux: bool = False, **kwargs):
        B = images.shape[0]

        sem_embeds   = self.sem_embeddings(images)    # [B, 1+R+N, C]
        recon_embeds = self.recon_embeddings(images)  # [B, 1+R+N, C]

        combined = torch.cat([sem_embeds, recon_embeds], dim=0)       # [2B, 1+R+N, C]
        hidden   = self.encoder(combined).last_hidden_state            # [2B, 1+R+N, C]
        hidden   = self.layernorm(hidden)

        # discard CLS + register tokens
        sem_out   = hidden[:B,  self._skip:]   # [B, N, C]
        recon_out = hidden[B:,  self._skip:]   # [B, N, C]

        sem_tokens   = self.sem_ln(sem_out)
        recon_tokens = self.recon_ln(self.recon_proj(recon_out))

        merged = torch.cat([sem_tokens, recon_tokens], dim=-1)

        if not return_aux:
            return merged

        return merged, {
            "sementic_tokens": sem_tokens,
            "recon_tokens":    recon_tokens,
            "merged_tokens":   merged,
        }

    def set_training_mode(self, mode: str = "full", **kwargs):
        self.requires_grad_(False)
        if mode == "full":
            self.requires_grad_(True)
        elif mode == "merger_recon_patchemb":
            for p in self.recon_embeddings.parameters():
                p.requires_grad = True
            for p in self.recon_proj.parameters():
                p.requires_grad = True
        elif mode == "frozen":
            pass
        else:
            raise ValueError(f"DINOv2Unified: unknown training mode '{mode}'")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DINOv2Unified set_training_mode='{mode}': "
            f"{trainable/1e6:.2f}M / {total/1e6:.2f}M trainable"
        )

    def get_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight

    def get_recon_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight
