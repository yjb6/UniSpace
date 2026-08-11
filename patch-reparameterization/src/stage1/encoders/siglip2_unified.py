import copy
import logging

import torch
from torch import nn
from transformers import SiglipModel

from . import register_encoder

logger = logging.getLogger(__name__)


@register_encoder()
class SigLIP2Unified(nn.Module):
    """Single-backbone SigLIP2 encoder with two independent embeddings.

    Architecture:
        image -> sem_embeddings   -> [B, N, C] --|
                                                  |--> cat [2B, N, C] -> encoder -> split
        image -> recon_embeddings -> [B, N, C] --|

        sem_out   -> sem_ln                          -> [B, N, C]
        recon_out -> recon_proj -> recon_ln          -> [B, N, recon_proj_dim]

        output: cat([sem, recon], dim=-1)            -> [B, N, C + recon_proj_dim]
    """

    def __init__(self, model_name: str, recon_proj_dim: int = 256, **kwargs):
        super().__init__()

        backbone = SiglipModel.from_pretrained(model_name).vision_model
        self.hidden_size = backbone.config.hidden_size
        self.patch_size = backbone.config.patch_size
        self.recon_proj_dim = recon_proj_dim

        # Two independent SiglipVisionEmbeddings (each has patch_embedding + position_embedding)
        self.sem_embeddings   = copy.deepcopy(backbone.embeddings)
        self.recon_embeddings = copy.deepcopy(backbone.embeddings)

        # Shared encoder
        self.encoder = backbone.encoder

        # recon projection + LN
        self.recon_proj = nn.Linear(self.hidden_size, recon_proj_dim, bias=True)
        nn.init.xavier_uniform_(self.recon_proj.weight)
        nn.init.zeros_(self.recon_proj.bias)

        self.sem_ln   = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.recon_ln = nn.LayerNorm(recon_proj_dim,   elementwise_affine=False)

        self.output_dim = self.hidden_size + recon_proj_dim
        self.recon_hidden_size = self.output_dim  # used by PatchReparam to set decoder latent_dim

        logger.info(
            f"SigLIP2Unified: hidden_size={self.hidden_size}, "
            f"recon_proj_dim={recon_proj_dim}, output_dim={self.output_dim}"
        )

    def forward(self, images: torch.Tensor, return_aux: bool = False, **kwargs):
        B = images.shape[0]

        sem_embeds   = self.sem_embeddings(images,   interpolate_pos_encoding=True)
        recon_embeds = self.recon_embeddings(images, interpolate_pos_encoding=True)

        combined    = torch.cat([sem_embeds, recon_embeds], dim=0)      # [2B, N, C]
        hidden      = self.encoder(inputs_embeds=combined).last_hidden_state  # [2B, N, C]

        sem_out   = hidden[:B]
        recon_out = hidden[B:]

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
            raise ValueError(f"SigLIP2Unified: unknown training mode '{mode}'")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SigLIP2Unified set_training_mode='{mode}': "
            f"{trainable/1e6:.2f}M / {total/1e6:.2f}M trainable"
        )

    def get_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight

    def get_recon_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight
