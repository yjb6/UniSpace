"""
Qwen3Unified: dual-path encoder using shared Qwen3-VL vision backbone.

Architecture mirrors SigLIP2Unified / DINOv2Unified:
  - Two independent patch embeddings (sem + recon), each a Qwen3VLVisionPatchEmbed
  - Shared backbone (transformer blocks + rotary pos emb + abs pos embed)
  - batch-concat along seq dim -> single backbone forward -> split
  - PatchMerger is skipped; outputs raw [N, hidden_size] tokens (no 4x compression)
  - recon branch: linear proj + LN
  - output: cat([sem_ln(sem_out), recon_ln(recon_proj(recon_out))], dim=-1)

Image preprocessing (caller's responsibility):
  Each image [C, H, W] must be converted to flat patches fed to patch_embed:
    pixel_values: [N * temporal_patch_size, C, patch_size, patch_size]
                = [H/p * W/p * 2, C, p, p]   (duplicate frame for temporal_patch_size=2)
  grid_thw: [num_images, 3]  with values [T=1, H/p, W/p]
            (T=1 because we duplicate the frame inside patch_embed, not pass 2 real frames)

  Concretely for one image of size (H, W) with patch_size=16:
    frame = image.unsqueeze(0).expand(2, -1, -1, -1)   # [2, C, H, W]
    patches = frame.unfold(2, 16, 16).unfold(3, 16, 16)
            .permute(2,3,0,1,4,5).reshape(-1, C, 2, 16, 16)  # [H/16*W/16, C, 2, 16, 16]
    pixel_values = patches.reshape(-1, C, 16, 16)            # [H/16*W/16*2, C, 16, 16]
    grid_thw = [[1, H//16, W//16]]

  For a batch, concat pixel_values and stack grid_thw along dim 0.
"""

import copy
import logging
from typing import Optional

import torch
from torch import nn

from . import register_encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import Qwen3VL vision components from transformers
# ---------------------------------------------------------------------------
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionBlock,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


@register_encoder()
class Qwen3Unified(nn.Module):
    """Shared-backbone Qwen3-VL vision encoder with two independent patch embeddings.

    PatchMerger is intentionally skipped so every spatial patch produces one token
    (no 2×2 spatial compression). Output token count = H/patch_size * W/patch_size.
    """

    def __init__(
        self,
        model_name: str,
        recon_proj_dim: int = 256,
        no_sem_ln: bool = False,
        use_deepstack: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Load full Qwen3VL model to extract the vision backbone, then discard the LLM
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        full_cfg = AutoConfig.from_pretrained(model_name)
        vis_cfg: Qwen3VLVisionConfig = full_cfg.vision_config

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu",
        )
        vis_model = full_model.visual
        full_model.model = None  # free LLM weights
        del full_model
        torch.cuda.empty_cache()

        self.hidden_size = vis_cfg.hidden_size          # 1152
        self.patch_size = vis_cfg.patch_size            # 16
        self.temporal_patch_size = vis_cfg.temporal_patch_size  # 2
        self.spatial_merge_size = vis_cfg.spatial_merge_size    # 2 (kept for rot_pos_emb compat)
        self.recon_proj_dim = recon_proj_dim

        # Two independent patch embeddings
        self.sem_patch_embed   = copy.deepcopy(vis_model.patch_embed)
        self.recon_patch_embed = copy.deepcopy(vis_model.patch_embed)

        # Two independent absolute position embeddings (learned, bilinear-interpolated)
        self.sem_pos_embed   = copy.deepcopy(vis_model.pos_embed)
        self.recon_pos_embed = copy.deepcopy(vis_model.pos_embed)
        self.num_grid_per_side = vis_model.num_grid_per_side

        # Shared: rotary pos emb + transformer blocks
        self.rotary_pos_emb = vis_model.rotary_pos_emb
        self.blocks = vis_model.blocks

        # deepstack: extract intermediate VIT features for injection into LLM early layers.
        # Only created when use_deepstack=True; old ckpts load fine via strict=False
        # (missing keys keep the Qwen pretrained init, existing keys load normally).
        self.use_deepstack = use_deepstack
        self.deepstack_visual_indexes = vis_cfg.deepstack_visual_indexes  # e.g. [8, 16, 24]
        if use_deepstack:
            self.deepstack_merger_list = copy.deepcopy(vis_model.deepstack_merger_list)

        # recon projection + LN
        self.recon_proj = nn.Linear(self.hidden_size, recon_proj_dim, bias=True)
        nn.init.xavier_uniform_(self.recon_proj.weight)
        nn.init.zeros_(self.recon_proj.bias)

        self.no_sem_ln = no_sem_ln
        self.sem_ln   = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.recon_ln = nn.LayerNorm(recon_proj_dim,   elementwise_affine=False)

        self.output_dim = self.hidden_size + recon_proj_dim
        self.recon_hidden_size = self.output_dim


        # Tell PatchReparam to skip its internal resize/normalize — we handle preprocessing externally
        self.handles_preprocessing = True
        # PatchReparam reads these to set encoder_mean/std and encoder_input_size
        import torch as _torch
        self.encoder_mean = _torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.encoder_std  = _torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.encoder_input_size = 256  # nominal value; actual size varies per bucket
        self.pixel_format = kwargs.get('pixel_format', 't_major')  # 'c_major' 与官方 Qwen3VL 一致

        del vis_model  # free memory

        logger.info(
            f"Qwen3Unified: hidden_size={self.hidden_size}, "
            f"recon_proj_dim={recon_proj_dim}, output_dim={self.output_dim}, "
            f"use_deepstack={use_deepstack}, "
            f"deepstack_indexes={self.deepstack_visual_indexes}, "
            f"pixel_format={self.pixel_format}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build 2D RoPE embeddings for a varlen batch."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row  = torch.arange(merge_size, device=device)
            intra_col  = torch.arange(merge_size, device=device)

            row_idx = (block_rows[:, None, None, None] * merge_size
                       + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * merge_size
                       + intra_col[None, None, None, :])

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            n = coords.shape[0]
            pos_ids[offset: offset + n] = coords
            offset += n

        return freq_table[pos_ids].flatten(1)  # [total_tokens, head_dim]

    def _abs_pos_embed(self, pos_embed_module: nn.Embedding, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear-interpolate absolute pos embeddings for a varlen batch."""
        N = self.num_grid_per_side
        idx_list    = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
            h_idxs = torch.linspace(0, N - 1, int(h))
            w_idxs = torch.linspace(0, N - 1, int(w))

            hf = h_idxs.int()
            wf = w_idxs.int()
            hc = (hf + 1).clamp(max=N - 1)
            wc = (wf + 1).clamp(max=N - 1)
            dh = h_idxs - hf
            dw = w_idxs - wf

            base_h  = hf  * N
            base_hc = hc * N

            indices = [
                (base_h[None].T  + wf[None]).flatten(),
                (base_h[None].T  + wc[None]).flatten(),
                (base_hc[None].T + wf[None]).flatten(),
                (base_hc[None].T + wc[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T       * (1 - dw)[None]).flatten(),
                (dh[None].T       * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        dev   = pos_embed_module.weight.device
        dtype = pos_embed_module.weight.dtype
        idx_t    = torch.tensor(idx_list,    dtype=torch.long,  device=dev)
        weight_t = torch.tensor(weight_list, dtype=dtype,        device=dev)

        # always call pos_embed_module — trainable, must participate in every forward
        embeds = pos_embed_module(idx_t)  # [4, N_total, D]
        result = (embeds * weight_t.unsqueeze(-1)).sum(0)  # [N_total, D]

        # repeat for temporal frames
        out = []
        offset = 0
        for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
            n = int(h * w)
            chunk = result[offset: offset + n]
            if t > 1:
                chunk = chunk.repeat(int(t), 1)
            out.append(chunk)
            offset += n
        return torch.cat(out, dim=0)

    def _block_order_indices(self, grid_thw: torch.Tensor, device=None) -> torch.Tensor:
        """Row-major -> 2x2 block order permutation indices."""
        merge = self.spatial_merge_size
        indices_all = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            assert h % merge == 0 and w % merge == 0
            mh, mw = h // merge, w // merge
            bh = torch.arange(mh, device=device)
            bw = torch.arange(mw, device=device)
            ih = torch.arange(merge, device=device)
            iw = torch.arange(merge, device=device)
            row = (bh[:, None, None, None] * merge + ih[None, None, :, None]).expand(mh, mw, merge, merge)
            col = (bw[None, :, None, None] * merge + iw[None, None, None, :]).expand(mh, mw, merge, merge)
            pf  = (row * w + col).reshape(-1)
            if t > 1:
                frame_offsets = torch.arange(t, device=device, dtype=torch.long) * (h * w)
                pf = (pf[None, :] + frame_offsets[:, None]).reshape(-1)
            indices_all.append(pf + offset)
            offset += t * h * w
        return torch.cat(indices_all, dim=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        return_aux: bool = False,
        **kwargs,
    ):
        """
        Args:
            pixel_values: [total_patches * temporal_patch_size, C, patch_size, patch_size]
                          flat patches for ALL images in the batch (sem and recon share input).
            grid_thw:     [num_images, 3]  — [T, H_grid, W_grid] per image.
                          T=1 for static images (temporal dim collapsed in patch_embed).
            return_aux:   if True, return (merged, aux_dict); else return merged.
        """
        # Patch embed (independent)
        sem_x   = self.sem_patch_embed(pixel_values)    # [total_tokens, D]
        recon_x = self.recon_patch_embed(pixel_values)  # [total_tokens, D]

        # Absolute pos embed (independent, bilinear interpolation)
        sem_pos   = self._abs_pos_embed(self.sem_pos_embed,   grid_thw)
        recon_pos = self._abs_pos_embed(self.recon_pos_embed, grid_thw)

        # Reorder from row-major to 2x2 block order to match _rot_pos_emb
        perm = self._block_order_indices(grid_thw, device=sem_x.device)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        sem_x   = (sem_x   + sem_pos)[perm]
        recon_x = (recon_x + recon_pos)[perm]

        # 2D RoPE (shared, already in 2x2 block order)
        rotary = self._rot_pos_emb(grid_thw)
        emb    = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens for flash attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        # Batch-concat along seq dim → one shared backbone forward
        seq_len = sem_x.shape[0]
        combined = torch.cat([sem_x, recon_x], dim=0)  # [2 * total_tokens, D]

        # cu_seqlens_2x: [0, n1, n1+n2, ..., seq_len, seq_len+n1, ...]
        cu_seqlens_2x = torch.cat([cu_seqlens, cu_seqlens[1:] + seq_len])

        pos_2x = (
            torch.cat([position_embeddings[0], position_embeddings[0]], dim=0),
            torch.cat([position_embeddings[1], position_embeddings[1]], dim=0),
        )

        deepstack_features = [] if self.use_deepstack else None

        for i, blk in enumerate(self.blocks):
            combined = blk(
                combined,
                cu_seqlens=cu_seqlens_2x,
                position_embeddings=pos_2x,
            )
            if self.use_deepstack and i in self.deepstack_visual_indexes:
                # Extract sem branch (block order) → row-major → merger projection
                # inv_perm converts block order back to row-major so the output
                # token order matches the main sem tokens in the LLM.
                sem_mid = combined[:seq_len][inv_perm]  # [N, hidden_size], row-major
                ds_idx  = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](sem_mid))

        sem_out   = combined[:seq_len][inv_perm]   # back to row-major
        recon_out = combined[seq_len:][inv_perm]

        sem_tokens   = sem_out if self.no_sem_ln else self.sem_ln(sem_out)
        recon_tokens = self.recon_ln(self.recon_proj(recon_out))
        merged = torch.cat([sem_tokens, recon_tokens], dim=-1)  # [total_tokens, D+recon_proj_dim]

        if not return_aux:
            if self.use_deepstack:
                return merged, deepstack_features
            return merged

        aux = {
            "sementic_tokens":   sem_tokens,
            "recon_tokens":      recon_tokens,
            "merged_tokens":     merged,
        }
        if self.use_deepstack:
            aux["deepstack_features"] = deepstack_features
        return merged, aux

    def set_training_mode(self, mode: str = "full", **kwargs):
        self.requires_grad_(False)
        if mode == "full":
            self.requires_grad_(True)
        elif mode == "merger_recon_patchemb":
            for p in self.recon_patch_embed.parameters():
                p.requires_grad = True
            for p in self.recon_pos_embed.parameters():
                p.requires_grad = True
            for p in self.recon_proj.parameters():
                p.requires_grad = True
        elif mode == "frozen":
            pass
        else:
            raise ValueError(f"Qwen3Unified: unknown training mode '{mode}'")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Qwen3Unified set_training_mode='{mode}': "
            f"{trainable/1e6:.2f}M / {total/1e6:.2f}M trainable"
        )

    def get_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight

    def get_recon_last_layer(self) -> nn.Parameter:
        return self.recon_proj.weight
