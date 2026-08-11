"""
Qwen3Frozen: 完全冻结的 Qwen3-VL vision backbone，只做 semantic encoding。
继承 Qwen3Unified 的所有辅助方法（RoPE、pos_embed、block_order），
去掉 recon 分支，输出 [total_tokens, 1152] 直接送给 decoder。
只训 decoder，encoder 完全不更新。
"""

import copy
import logging

import torch
from torch import nn

from . import register_encoder
from .qwen3_unified import Qwen3Unified

logger = logging.getLogger(__name__)


@register_encoder()
class Qwen3Frozen(Qwen3Unified):
    """
    完全冻结的单路 Qwen3-VIT encoder（无 recon 分支）。
    继承 Qwen3Unified 的所有辅助方法，去掉双路设计。
    输出：[total_tokens, hidden_size=1152]
    """

    def __init__(self, model_name: str, **kwargs):
        # 不调用 Qwen3Unified.__init__，自己初始化
        nn.Module.__init__(self)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )
        from transformers import AutoConfig

        full_cfg = AutoConfig.from_pretrained(model_name)
        vis_cfg  = full_cfg.vision_config

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        vis_model = full_model.visual
        del full_model.model

        self.hidden_size         = vis_cfg.hidden_size
        self.patch_size          = vis_cfg.patch_size
        self.temporal_patch_size = vis_cfg.temporal_patch_size
        self.spatial_merge_size  = vis_cfg.spatial_merge_size
        self.recon_proj_dim      = 0  # no recon branch

        # single patch embed + pos embed (no deepcopy needed, frozen anyway)
        self.sem_patch_embed   = vis_model.patch_embed
        self.sem_pos_embed     = vis_model.pos_embed
        self.num_grid_per_side = vis_model.num_grid_per_side
        self.rotary_pos_emb    = vis_model.rotary_pos_emb
        self.blocks            = vis_model.blocks

        self.output_dim        = self.hidden_size
        self.recon_hidden_size = self.hidden_size  # decoder latent_dim

        # PatchReparam reads these to skip internal resize/normalize
        self.handles_preprocessing = True
        self.encoder_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.encoder_std  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.encoder_input_size = 256

        # caches (inherited methods use these)
        self._pos_embed_cache:   dict = {}
        self._rot_pos_emb_cache: dict = {}
        self._block_idx_cache:   dict = {}

        # 完全冻结
        self.requires_grad_(False)

        del vis_model

        logger.info(f"Qwen3Frozen: hidden_size={self.hidden_size}, fully frozen")

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        return_aux: bool = False,
        **kwargs,
    ):
        # patch embed + pos embed (single path)
        x = self.sem_patch_embed(pixel_values)
        pos = self._abs_pos_embed(self.sem_pos_embed, grid_thw)

        # reorder to 2x2 block order
        perm = self._block_order_indices(grid_thw, device=x.device)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
        x = (x + pos)[perm]

        # RoPE
        rotary = self._rot_pos_emb(grid_thw)
        emb    = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        # blocks forward
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        # restore row-major
        x = x[inv_perm]

        if not return_aux:
            return x

        return x, {"sementic_tokens": x, "merged_tokens": x}

    def set_training_mode(self, mode: str = "frozen", **kwargs):
        self.requires_grad_(False)
        logger.info(f"Qwen3Frozen: always frozen, ignoring mode='{mode}'")

    def get_last_layer(self):
        return None

    def get_recon_last_layer(self):
        return None
