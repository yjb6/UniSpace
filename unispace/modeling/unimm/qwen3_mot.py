# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# MOT (Mixture-of-Transformers) version of Qwen3 (text-only LLM) model.
# Key difference from Transfusion: und/gen tokens use independent parameters
# (QKV projections, LayerNorm, MLP) but share attention computation.
# Differs from qwen3_vl_mot.py: uses 1D RoPE, no vision model, flat config.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

# flex_attention: GPU (PyTorch 2.5+) only
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

# flex_attention 调度:
# - 如果外层 decoder layer 已被 torch.compile，直接用原始 flex_attention
#   （外层 compile 会生成完整 Triton kernel，嵌套 compile 可能导致 NaN）
# - 如果外层未 compile，用 _compiled_flex_attention（函数级 compile）
from .qwen3 import _compiled_flex_attention

# Import shared components from qwen3.py (text-only LLM version)
from .qwen3 import (
    Qwen3TextRMSNorm,
    Qwen3TextMLP,
    Qwen3TextRotaryEmbedding,
    Qwen3TextSDPAAttention,
    NaiveCache,
    Qwen3PreTrainedModel,
    Qwen3Model,
    Qwen3ForConditionalGeneration,
    apply_rotary_pos_emb,
    repeat_kv,
    universal_varlen_attention,
    pad_sequence,
    ATTENTION_BACKEND,
)
from modeling.qwen3.configuration_qwen3 import Qwen3Config
from transformers.cache_utils import Cache

def _scatter_at(dest: torch.Tensor, idx: torch.LongTensor, src: torch.Tensor) -> torch.Tensor:
    """
    Assign ``dest[idx] = src`` with torch.compile compatibility.

    When called inside a compiled graph (dynamo tracing), uses out-of-place
    ``scatter`` to avoid the index_put_ backward NaN bug on PyTorch 2.5.x.

    When running eagerly (NPU / non-compiled GPU), uses plain index assignment
    which is well-tested across all backends.
    """
    if is_torchdynamo_compiling():
        # Out-of-place scatter: compile-safe, correct backward in Triton
        shape = [idx.shape[0]] + [1] * (src.ndim - 1)
        idx_expanded = idx.view(shape).expand_as(src)
        return dest.scatter(0, idx_expanded, src)
    else:
        # Inplace index_put_: standard path, reliable on NPU/GPU eager
        dest[idx] = src
        return dest

# ============================================================================
# MOT Attention: Independent QKV/O projections for und/gen, shared attention
# ============================================================================

class Qwen3TextMoTSDPAAttention(nn.Module):
    """
    MOT version of SDPA attention.
    und tokens and gen tokens use independent QKV/O projections and QK norms,
    but share the attention computation (SDPA).
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False

        # === UND branch (shared with original transfusion weights) ===
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3TextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # === GEN branch (MOT-specific independent parameters) ===
        self.q_proj_moe_gen = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm_moe_gen = Qwen3TextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen3TextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        """Dispatch to train/inference based on mode."""
        mode = kwargs.get('mode', 'und')
        is_causal = kwargs.get('is_causal', True)
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_inference(**kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: torch.Tensor,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence.shape[0]
        freeze_und = getattr(self.config, 'freeze_und', False)
        compute_dtype = self.q_proj.weight.dtype

        # 1. Create empty tensors for packed QKV states (use projection weight dtype)
        packed_query_states = torch.zeros(
            total_seq_len, self.num_heads * self.head_dim,
            device=packed_sequence.device, dtype=compute_dtype)
        packed_key_states = torch.zeros(
            total_seq_len, self.num_key_value_heads * self.head_dim,
            device=packed_sequence.device, dtype=compute_dtype)
        packed_value_states = torch.zeros(
            total_seq_len, self.num_key_value_heads * self.head_dim,
            device=packed_sequence.device, dtype=compute_dtype)

        # 2. Route und tokens through und QKV projections
        # Use _scatter_at (out-of-place scatter) instead of inplace index_put_
        # to avoid NaN in compiled backward on PyTorch 2.5.x
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            packed_sequence_und = packed_sequence[packed_und_token_indexes]
            packed_query_states = _scatter_at(packed_query_states, packed_und_token_indexes, self.q_proj(packed_sequence_und))
            packed_key_states = _scatter_at(packed_key_states, packed_und_token_indexes, self.k_proj(packed_sequence_und))
            packed_value_states = _scatter_at(packed_value_states, packed_und_token_indexes, self.v_proj(packed_sequence_und))

        # 3. Route gen tokens through gen QKV projections
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
            packed_query_states = _scatter_at(packed_query_states, packed_gen_token_indexes, self.q_proj_moe_gen(packed_sequence_gen))
            packed_key_states = _scatter_at(packed_key_states, packed_gen_token_indexes, self.k_proj_moe_gen(packed_sequence_gen))
            packed_value_states = _scatter_at(packed_value_states, packed_gen_token_indexes, self.v_proj_moe_gen(packed_sequence_gen))

        # 4. Reshape to (seq_len, num_heads, head_dim)
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # 5. freeze_und: detach value states for und tokens
        if freeze_und and packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            packed_value_states = _scatter_at(
                packed_value_states, packed_und_token_indexes,
                packed_value_states[packed_und_token_indexes].detach())

        # 6. QK Norm - route through separate norms
        # Use _scatter_at for compile-safe out-of-place scatter
        normed_query_states = packed_query_states.new_zeros(packed_query_states.shape)
        normed_key_states = packed_key_states.new_zeros(packed_key_states.shape)

        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_q_normed = self.q_norm(packed_query_states[packed_und_token_indexes])
            und_k_normed = self.k_norm(packed_key_states[packed_und_token_indexes])
            if freeze_und:
                und_q_normed = und_q_normed.detach()
                und_k_normed = und_k_normed.detach()
            normed_query_states = _scatter_at(normed_query_states, packed_und_token_indexes, und_q_normed)
            normed_key_states = _scatter_at(normed_key_states, packed_und_token_indexes, und_k_normed)

        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            normed_query_states = _scatter_at(normed_query_states, packed_gen_token_indexes,
                self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes]))
            normed_key_states = _scatter_at(normed_key_states, packed_gen_token_indexes,
                self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes]))

        # NOTE: Do NOT detach gen K/V here.
        # Bagel's original MoT does not detach gen K/V. Detaching would cut ALL
        # gradients to k_proj_moe_gen / v_proj_moe_gen, making them untrainable
        # and preventing the model from learning text-conditioned generation.

        # 7. Shared RoPE
        packed_cos, packed_sin = packed_position_embeddings
        packed_cos = packed_cos.squeeze(0)
        packed_sin = packed_sin.squeeze(0)
        normed_query_states, normed_key_states = apply_rotary_pos_emb(
            normed_query_states, normed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        # 8. Shared attention — dual-backend dispatch
        attn_impl = self.config._attn_implementation

        # ── flex_attention 路径: 使用 BlockMask（GPU 高效稀疏 attention）──
        if attn_impl == "flex_attention":
            assert flex_attention is not None, (
                "flex_attention not available. Use --attn_implementation sdpa"
            )
            pad_size = total_seq_len - normed_query_states.shape[0]
            _packed_q = pad_sequence(normed_query_states.permute(1, 0, 2), pad_size)
            _packed_k = pad_sequence(normed_key_states.permute(1, 0, 2), pad_size)
            _packed_v = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            # 当外层 layer 已被 torch.compile 时，用原始 flex_attention 避免嵌套 compile
            # (is_torchdynamo_compiling() 在 dynamo trace 时返回 True)
            _flex_fn = flex_attention if is_torchdynamo_compiling() else _compiled_flex_attention
            packed_attn_output = _flex_fn(
                _packed_q.unsqueeze(0),
                _packed_k.unsqueeze(0),
                _packed_v.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]
            packed_attn_output = packed_attn_output.transpose(
                0, 1).reshape(-1, self.num_heads * self.head_dim)

        # ── SDPA 路径（默认）: 使用稠密 bool mask ──
        else:
            query_states = normed_query_states.view(
                total_seq_len, self.num_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)
            key_states = normed_key_states.view(
                total_seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)
            value_states = packed_value_states.view(
                total_seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get(attn_impl)
            if attention_interface is not None:
                attn_output, _ = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling,
                    is_causal=False,
                )
                packed_attn_output = attn_output.squeeze(0).reshape(
                    -1, self.num_heads * self.head_dim)
            else:
                # Fallback: 手动 repeat_kv + SDPA
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16),
                    key_states.to(torch.bfloat16),
                    value_states.to(torch.bfloat16),
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
                packed_attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(
                    -1, self.num_heads * self.head_dim)

        # 9. O projection - route through separate output projections
        # Cast to projection weight dtype to avoid dtype mismatch
        packed_attn_output = packed_attn_output.to(compute_dtype)
        output = torch.zeros(
            total_seq_len, self.hidden_size,
            device=packed_attn_output.device, dtype=compute_dtype)
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_output = self.o_proj(packed_attn_output[packed_und_token_indexes])
            if freeze_und:
                und_output = und_output.detach()
            output = _scatter_at(output, packed_und_token_indexes, und_output)

        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            output = _scatter_at(output, packed_gen_token_indexes,
                self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes]))

        return output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        attention_mask=None,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        """
        Inference forward.
        mode='und': all tokens go through und parameters (same as transfusion).
        mode='gen': text tokens go through und params, vae tokens go through gen params.
        """
        total_len = packed_query_sequence.shape[0]

        # Determine the compute dtype from projection weights to avoid dtype mismatch
        compute_dtype = self.q_proj.weight.dtype

        if mode == "und":
            # UND mode: all tokens use und parameters (identical to transfusion)
            packed_query_states = self.q_proj(
                packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(
                packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(
                packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        else:
            # GEN mode: route text tokens to und params, vae tokens to gen params
            packed_query_states = torch.zeros(
                total_len, self.num_heads * self.head_dim,
                device=packed_query_sequence.device, dtype=compute_dtype)
            packed_key_states = torch.zeros(
                total_len, self.num_key_value_heads * self.head_dim,
                device=packed_query_sequence.device, dtype=compute_dtype)
            packed_value_states = torch.zeros(
                total_len, self.num_key_value_heads * self.head_dim,
                device=packed_query_sequence.device, dtype=compute_dtype)

            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                text_seq = packed_query_sequence[packed_text_indexes]
                packed_query_states[packed_text_indexes] = self.q_proj(text_seq)
                packed_key_states[packed_text_indexes] = self.k_proj(text_seq)
                packed_value_states[packed_text_indexes] = self.v_proj(text_seq)

            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                vae_seq = packed_query_sequence[packed_vae_token_indexes]
                packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(vae_seq)
                packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(vae_seq)
                packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(vae_seq)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            # QK norm routing
            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                packed_query_states[packed_text_indexes] = self.q_norm(
                    packed_query_states[packed_text_indexes])
                packed_key_states[packed_text_indexes] = self.k_norm(
                    packed_key_states[packed_text_indexes])
            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                    packed_query_states[packed_vae_token_indexes])
                packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                    packed_key_states[packed_vae_token_indexes])

        # RoPE
        packed_cos, packed_sin = packed_query_position_embeddings
        packed_cos = packed_cos.squeeze(0)
        packed_sin = packed_sin.squeeze(0)
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        # KV Cache merge
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]
            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        # Flash Attention
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = universal_varlen_attention(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
            dropout_p=0.0,
        )

        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
        # Cast to projection weight dtype to avoid dtype mismatch
        packed_attn_output = packed_attn_output.to(compute_dtype)

        # O projection routing
        if mode == "und":
            packed_attn_output = self.o_proj(packed_attn_output)
        else:
            output = torch.zeros(
                total_len, self.hidden_size,
                device=packed_attn_output.device, dtype=compute_dtype)
            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                    packed_attn_output[packed_vae_token_indexes])
            packed_attn_output = output

        # Update KV Cache
        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


# ============================================================================
# MOT Decoder Layer: Independent LayerNorms + MLP for und/gen
# ============================================================================

class Qwen3TextMoTDecoderLayer(nn.Module):
    """
    MOT version of decoder layer.
    und/gen tokens use independent LayerNorms and MLPs,
    and share the MOT attention (which internally routes QKV/O).
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        # MOT attention (handles und/gen routing internally)
        self.self_attn = Qwen3TextMoTSDPAAttention(config=config, layer_idx=layer_idx)

        # === UND branch ===
        self.mlp = Qwen3TextMLP(config)
        self.input_layernorm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # === GEN branch (MOT-specific) ===
        self.mlp_moe_gen = Qwen3TextMLP(config)
        self.input_layernorm_moe_gen = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list,
        attention_mask: torch.Tensor,
        packed_position_embeddings: tuple,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        freeze_und = getattr(self.config, 'freeze_und', False)

        # === Pre-attention LayerNorm routing ===
        residual = packed_sequence
        normed_sequence = torch.zeros_like(packed_sequence)

        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            normed_sequence = _scatter_at(normed_sequence, packed_und_token_indexes,
                self.input_layernorm(packed_sequence[packed_und_token_indexes]))

        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            normed_sequence = _scatter_at(normed_sequence, packed_gen_token_indexes,
                self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes]))

        # === MOT Attention (routing handled inside) ===
        packed_sequence = self.self_attn(
            packed_sequence=normed_sequence,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

        # freeze_und: detach attention output for und tokens
        if freeze_und and packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            packed_sequence = _scatter_at(packed_sequence, packed_und_token_indexes,
                packed_sequence[packed_und_token_indexes].detach())

        packed_sequence = residual + packed_sequence

        # === Post-attention LayerNorm + MLP routing ===
        residual = packed_sequence
        mlp_output = torch.zeros_like(packed_sequence)

        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_normed = self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
            und_mlp_out = self.mlp(und_normed)
            if freeze_und:
                und_mlp_out = und_mlp_out.detach()
            mlp_output = _scatter_at(mlp_output, packed_und_token_indexes, und_mlp_out)

        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            gen_normed = self.post_attention_layernorm_moe_gen(
                packed_sequence[packed_gen_token_indexes])
            mlp_output = _scatter_at(mlp_output, packed_gen_token_indexes,
                self.mlp_moe_gen(gen_normed))

        packed_sequence = residual + mlp_output

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple,
        packed_query_indexes: torch.Tensor,
        attention_mask=None,
        past_key_values: Optional[Any] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> tuple:
        if mode == "und":
            # UND mode: standard single path (same as transfusion)
            residual = packed_query_sequence
            packed_query_sequence = self.input_layernorm(packed_query_sequence)

            packed_query_sequence, past_key_values = self.self_attn(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                attention_mask=attention_mask,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
            )
            packed_query_sequence = residual + packed_query_sequence

            residual = packed_query_sequence
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
            packed_query_sequence = residual + packed_query_sequence
        else:
            # GEN mode: route text to und params, vae to gen params
            total_len = packed_query_sequence.shape[0]
            compute_dtype = self.input_layernorm.weight.dtype
            residual = packed_query_sequence.to(compute_dtype)
            normed_sequence = torch.zeros(
                total_len, packed_query_sequence.shape[-1],
                device=packed_query_sequence.device, dtype=compute_dtype)

            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                normed_sequence[packed_text_indexes] = self.input_layernorm(
                    residual[packed_text_indexes])
            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                normed_sequence[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                    residual[packed_vae_token_indexes])

            packed_query_sequence, past_key_values = self.self_attn(
                packed_query_sequence=normed_sequence,
                query_lens=query_lens,
                attention_mask=attention_mask,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
            packed_query_sequence = residual + packed_query_sequence

            residual = packed_query_sequence
            mlp_output = torch.zeros_like(packed_query_sequence)

            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                text_normed = self.post_attention_layernorm(
                    packed_query_sequence[packed_text_indexes])
                mlp_output[packed_text_indexes] = self.mlp(text_normed)
            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                vae_normed = self.post_attention_layernorm_moe_gen(
                    packed_query_sequence[packed_vae_token_indexes])
                mlp_output[packed_vae_token_indexes] = self.mlp_moe_gen(vae_normed)

            packed_query_sequence = residual + mlp_output

        return packed_query_sequence, past_key_values


# ============================================================================
# MOT Text Model: Uses MoTDecoderLayer + dual final norm
# ============================================================================

class Qwen3TextMoTModel(Qwen3PreTrainedModel):
    """
    MOT version of Qwen3TextModel.
    Uses Qwen3TextMoTDecoderLayer and adds a separate norm for gen tokens.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3TextMoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # UND final norm
        self.norm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GEN final norm (MOT-specific)
        self.norm_moe_gen = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3TextRotaryEmbedding(config=config)
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        hidden_states = packed_sequence

        # 1D RoPE: ensure position_ids is [1, seq_len]
        # Data pipeline may provide 3D M-RoPE format [3, seq_len] — extract first dim
        # (for text tokens, all 3 dims are identical)
        if packed_position_ids.ndim == 2 and packed_position_ids.shape[0] == 3:
            packed_position_ids = packed_position_ids[0]  # [3, seq_len] → [seq_len]
        if packed_position_ids.ndim == 1:
            packed_position_ids = packed_position_ids.unsqueeze(0)  # [seq_len] → [1, seq_len]

        position_embeddings = self.rotary_emb(hidden_states, packed_position_ids)

        # When using flex_attention, sample_lens is NOT used inside decoder layers
        # (the BlockMask already encodes the attention pattern).  Passing the raw
        # Python list would cause torch.compile to guard on its contents, triggering
        # recompilation whenever the number/sizes of packed samples change.  We pass
        # None to eliminate this guard.  For SDPA, sample_lens is required.
        attn_impl = getattr(self.config, '_attn_implementation', 'sdpa')
        _sample_lens = None if attn_impl == 'flex_attention' else sample_lens

        # Through all MOT decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                packed_sequence=hidden_states,
                sample_lens=_sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        # Final norm: route und/gen to separate norms
        norm_dtype = self.norm.weight.dtype
        hidden_states = hidden_states.to(norm_dtype)
        output = torch.zeros_like(hidden_states)
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            output = _scatter_at(output, packed_und_token_indexes,
                self.norm(hidden_states[packed_und_token_indexes]))
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            output = _scatter_at(output, packed_gen_token_indexes,
                self.norm_moe_gen(hidden_states[packed_gen_token_indexes]))

        return output

    def forward_inference(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[Any] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> tuple:
        hidden_states = packed_sequence

        # 1D RoPE: ensure position_ids is [1, seq_len]
        # Data pipeline may provide 3D M-RoPE format [3, seq_len] — extract first dim
        # (for text tokens, all 3 dims are identical)
        if packed_position_ids.ndim == 2 and packed_position_ids.shape[0] == 3:
            packed_position_ids = packed_position_ids[0]  # [3, seq_len] → [seq_len]
        if packed_position_ids.ndim == 1:
            packed_position_ids = packed_position_ids.unsqueeze(0)  # [seq_len] → [1, seq_len]

        position_embeddings = self.rotary_emb(hidden_states, packed_position_ids)

        extra_inputs = {}
        extra_inputs.update(mode=mode)
        if mode == 'gen':
            assert packed_vae_token_indexes is not None
            assert packed_text_indexes is not None
            extra_inputs.update(
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )

        for decoder_layer in self.layers:
            hidden_states, past_key_values = decoder_layer(
                packed_query_sequence=hidden_states,
                query_lens=sample_lens,
                attention_mask=attention_mask,
                packed_query_position_embeddings=position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        # Final norm routing
        if mode == "und":
            hidden_states = self.norm(hidden_states)
        else:
            norm_dtype = self.norm.weight.dtype
            hidden_states = hidden_states.to(norm_dtype)
            output = torch.zeros_like(hidden_states)
            if packed_text_indexes is not None and len(packed_text_indexes) > 0:
                output[packed_text_indexes] = self.norm(hidden_states[packed_text_indexes])
            if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                output[packed_vae_token_indexes] = self.norm_moe_gen(
                    hidden_states[packed_vae_token_indexes])
            hidden_states = output

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



# ============================================================================
# MOT Model (wraps language only, no vision): Uses Qwen3TextMoTModel
# ============================================================================

class Qwen3MoTModel(Qwen3Model):
    """
    MOT version of Qwen3Model (text-only, no VIT).
    Overrides language_model to use Qwen3TextMoTModel.
    No vision processing — VIT is handled externally by Unimm (SigLIP + connector).
    """

    def __init__(self, config):
        # Skip Qwen3Model.__init__ to avoid creating non-MOT language_model
        Qwen3PreTrainedModel.__init__(self, config)
        # No self.visual — Qwen3 (text-only) uses external SigLIP VIT
        # Use MOT text model instead of standard text model
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.language_model = Qwen3TextMoTModel._from_config(text_config)
        self.post_init()


# ============================================================================
# MOT Conditional Generation Model: Top-level model
# ============================================================================

class Qwen3MoTForConditionalGeneration(Qwen3ForConditionalGeneration):
    """
    MOT version of Qwen3ForConditionalGeneration.
    Uses Qwen3MoTModel and adds init_moe() for parameter initialization.
    """

    def __init__(self, config):
        # Ensure text_config compatibility for flat Qwen3Config
        if not hasattr(config, 'text_config'):
            config.text_config = config
        # Skip parent __init__ to use MOT model
        Qwen3PreTrainedModel.__init__(self, config)
        GenerationMixin.__init__(self)
        self.model = Qwen3MoTModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def init_moe(self):
        """
        Copy und parameters to gen parameters for initialization from
        a transfusion checkpoint. This allows MOT model to start training
        from an existing transfusion model.
        """
        param_dict = dict(self.named_parameters())
        copied_count = 0
        for name, param in list(param_dict.items()):
            if "moe_gen" in name:
                # Map gen parameter back to und parameter
                original_name = name.replace("_moe_gen", "")
                if original_name in param_dict:
                    param.data.copy_(param_dict[original_name].data)
                    copied_count += 1
        print(f"[init_moe] Copied {copied_count} parameter tensors from und to gen branch.")
        return copied_count


__all__ = [
    "Qwen3TextMoTSDPAAttention",
    "Qwen3TextMoTDecoderLayer",
    "Qwen3TextMoTModel",
    "Qwen3MoTModel",
    "Qwen3MoTForConditionalGeneration",
]
