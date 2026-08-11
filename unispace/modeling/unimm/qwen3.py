# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
#
# Adapted from modeling/qwen3/modeling_qwen3.py for unimm packed training.
# Key changes vs original Qwen3:
#   - Relative imports → absolute imports (transformers.xxx)
#   - Attention → Qwen3TextSDPAAttention (packed training + inference, flex/sdpa)
#   - DecoderLayer → Qwen3TextDecoderLayer (forward_train / forward_inference)
#   - Model → Qwen3TextModel (1D RoPE, no vision)
#   - Top-level → Qwen3Model + Qwen3ForConditionalGeneration (no VIT, compatible with Unimm)

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
import os
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

# flex_attention: GPU (PyTorch 2.5+) 支持，NPU 不支持，条件导入
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

# torch.compile 包装的 flex_attention，必须在模块级别定义
_compiled_flex_attention = torch.compile(flex_attention) if flex_attention is not None else None

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs as _check_model_inputs_raw

# 兼容 transformers 4.57.1（plain decorator）和 4.57.3+（factory with args）
if 'tie_last_hidden_states' in inspect.signature(_check_model_inputs_raw).parameters:
    check_model_inputs = _check_model_inputs_raw()
else:
    check_model_inputs = _check_model_inputs_raw

from modeling.qwen3.configuration_qwen3 import Qwen3Config

# ── Attention backend detection ──
def get_attention_backend():
    env_backend = os.environ.get('ATTENTION_BACKEND', '').strip().lower()
    if env_backend in ('flash_attn', 'torch_sdpa'):
        return env_backend
    try:
        import flash_attn  # noqa: F401
        return 'flash_attn'
    except ImportError:
        return 'torch_sdpa'

ATTENTION_BACKEND = get_attention_backend()

if ATTENTION_BACKEND == 'flash_attn':
    from flash_attn import flash_attn_varlen_func
    print("[Qwen3 Attention] Using Flash Attention for variable-length sequences")
else:
    print("[Qwen3 Attention] Using PyTorch SDPA fallback")

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096


# ── Variable-length attention ──
def universal_varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    if ATTENTION_BACKEND == 'flash_attn':
        return flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            causal=causal, dropout_p=dropout_p,
        )
    # SDPA fallback
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    if num_heads != num_kv_heads:
        num_groups = num_heads // num_kv_heads
        k = k[:, :, None, :].expand(-1, -1, num_groups, -1).reshape(-1, num_heads, head_dim)
        v = v[:, :, None, :].expand(-1, -1, num_groups, -1).reshape(-1, num_heads, head_dim)
    q_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    k_seqlens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    q_list = q.split(q_seqlens, dim=0)
    k_list = k.split(k_seqlens, dim=0)
    v_list = v.split(k_seqlens, dim=0)
    outputs = []
    for qi, ki, vi in zip(q_list, k_list, v_list):
        q_len_i, k_len_i = qi.shape[0], ki.shape[0]
        qi = qi.permute(1, 0, 2).unsqueeze(0)
        ki = ki.permute(1, 0, 2).unsqueeze(0)
        vi = vi.permute(1, 0, 2).unsqueeze(0)
        if causal and q_len_i != k_len_i:
            past_len = k_len_i - q_len_i
            attn_mask = qi.new_zeros(1, 1, q_len_i, k_len_i)
            causal_part = torch.triu(
                torch.full((q_len_i, q_len_i), float('-inf'), device=qi.device, dtype=qi.dtype), diagonal=1
            )
            attn_mask[0, 0, :, past_len:past_len + q_len_i] = causal_part
            attn_out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None, dropout_p=dropout_p, is_causal=causal)
        attn_out = attn_out.squeeze(0).permute(1, 0, 2)
        outputs.append(attn_out)
    return torch.cat(outputs, dim=0)


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


# ── NaiveCache (KV cache for inference) ──
class NaiveCache:
    """简单的 KV 缓存实现,兼容 transformers Cache 接口"""
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
        self._seen_tokens = 0
        self._num_layers = num_layers

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0

    def get_seq_length(self, layer_idx=0):
        if self.key_cache.get(layer_idx) is not None:
            return self.key_cache[layer_idx].shape[0]
        return 0

    def get_max_length(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        if key_states is not None:
            self._seen_tokens = key_states.shape[0]
        return key_states, value_states

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def reset(self):
        for k in self.key_cache:
            self.key_cache[k] = None
            self.value_cache[k] = None
        self._seen_tokens = 0


# ── RMSNorm ──
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Alias for weight loading compatibility
Qwen3RMSNorm = Qwen3TextRMSNorm


# ── MLP ──
class Qwen3TextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# Alias for weight loading compatibility
Qwen3MLP = Qwen3TextMLP


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3TextSDPAAttention(nn.Module):
    """
    Qwen3 attention adapted for packed training/inference.
    Supports flex_attention (GPU) and SDPA backends.
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
        self.is_causal = False  # packed training: mask controls causality

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

    def forward(self, *args, **kwargs):
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
        packed_und_token_indexes: torch.LongTensor = None,
        packed_gen_token_indexes: torch.LongTensor = None,
    ):
        # QKV projection
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        # QK Norm
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        # RoPE
        packed_cos, packed_sin = packed_position_embeddings
        packed_cos = packed_cos.squeeze(0)
        packed_sin = packed_sin.squeeze(0)
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        total_seq_len = sum(sample_lens)
        attn_impl = self.config._attn_implementation

        # ── flex_attention path ──
        if attn_impl == "flex_attention":
            assert flex_attention is not None, "flex_attention not available"
            pad_size = total_seq_len - packed_query_states.shape[0]
            _packed_q = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            _packed_k = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            _packed_v = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = _compiled_flex_attention(
                _packed_q.unsqueeze(0), _packed_k.unsqueeze(0), _packed_v.unsqueeze(0),
                enable_gqa=True, block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]
            packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)

        # ── SDPA 路径（默认）: 使用稠密 bool mask ──
        # NPU: bool mask 触发 FlashAttentionScore 高性能 kernel（不 materialize S×S 矩阵）
        # GPU: bool mask 走 SDPA math/efficient kernel
        else:
            query_states = packed_query_states.view(
                total_seq_len, self.num_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)
            key_states = packed_key_states.view(
                total_seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)
            value_states = packed_value_states.view(
                total_seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(0, 1).unsqueeze(0)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get(attn_impl)
            if attention_interface is not None:
                attn_output, _ = attention_interface(
                    self, query_states, key_states, value_states,
                    attention_mask=attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling, is_causal=False,
                )
                packed_attn_output = attn_output.squeeze(0).reshape(-1, self.num_heads * self.head_dim)
            else:
                # Fallback: 手动 repeat_kv + SDPA
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16), key_states.to(torch.bfloat16),
                    value_states.to(torch.bfloat16), attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False,
                )
                packed_attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)

        packed_attn_output = self.o_proj(packed_attn_output)
        return packed_attn_output

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
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

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
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = F.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = F.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = universal_varlen_attention(
            q=packed_query_states, k=merged_key_states, v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(query_lens).item(), max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal, dropout_p=0.0,
        )

        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

# Alias for weight loading: original Qwen3 uses "Qwen3Attention" as module name
Qwen3Attention = Qwen3TextSDPAAttention


class Qwen3TextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen3TextSDPAAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3TextMLP(config)
        self.input_layernorm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Safe access to layer_types
        if hasattr(config, 'layer_types') and config.layer_types:
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"

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
        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        packed_sequence = residual + packed_sequence

        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence
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
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values

# Alias for weight loading compatibility
Qwen3DecoderLayer = Qwen3TextDecoderLayer


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3TextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3TextDecoderLayer,
        "attentions": Qwen3TextSDPAAttention,
    }


class Qwen3TextRotaryEmbedding(nn.Module):
    """Standard 1D RoPE for Qwen3 (not 3D M-RoPE like Qwen3-VL)."""
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # 1D RoPE: position_ids shape [batch, seq_len]
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Alias for weight loading compatibility
Qwen3RotaryEmbedding = Qwen3TextRotaryEmbedding


class Qwen3TextModel(Qwen3PreTrainedModel):
    """Text-only model (no vision). Analogous to Qwen3VLTextModel but with 1D RoPE."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                packed_sequence=hidden_states,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

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

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

# Keep original class name as alias for weight loading
Qwen3Model_Original = Qwen3TextModel


# ── Top-level model: Qwen3Model (no VIT, analogous to Qwen3VLModel) ──
class Qwen3Model(Qwen3PreTrainedModel):
    """
    Qwen3 model wrapper (no vision). Structure:
      Qwen3Model.embed_tokens  (via language_model)
      Qwen3Model.language_model = Qwen3TextModel
    This mirrors Qwen3VLModel which has .visual + .language_model.
    For Qwen3 (text-only), there is no .visual.
    """
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        # For Qwen3 flat config, text_config points to self
        if not hasattr(config, 'text_config'):
            config.text_config = config
        self.language_model = Qwen3TextModel._from_config(config)
        self.post_init()

    # Convenience accessors (match Qwen3VLModel interface)
    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

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
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.language_model(
            packed_sequence=packed_sequence,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_sequence: torch.Tensor = None,
        sample_lens: list = None,
        attention_mask=None,
        packed_position_ids: torch.Tensor = None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.language_model(
            packed_sequence=packed_sequence if packed_sequence is not None else inputs_embeds,
            packed_position_ids=packed_position_ids,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )


# ── Top-level: Qwen3ForConditionalGeneration ──
class Qwen3ForConditionalGeneration(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 for conditional generation (text-only LLM backbone).
    Structure: self.model = Qwen3Model, self.lm_head
    Compatible with Unimm interface:
      - self.model.language_model.embed_tokens  → embedding
      - self.model.embed_tokens                 → shortcut
      - self.lm_head                            → output head
    """
    # HF Qwen3 checkpoint keys: model.layers.*, model.embed_tokens.*, lm_head.*
    # Our structure: model.language_model.layers.*, model.language_model.embed_tokens.*, lm_head.*
    # Map "model.X" → "model.language_model.X" during weight loading
    _checkpoint_conversion_mapping = {"^model\\.": "model.language_model."}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3Config

    def __init__(self, config):
        # Ensure text_config self-reference for flat Qwen3Config
        if not hasattr(config, 'text_config'):
            config.text_config = config
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    # Convenience: match Qwen3VLForConditionalGeneration interface
    @property
    def language_model(self):
        return self.model.language_model

    @check_model_inputs
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
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        sample_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        key_values_lens=None,
        packed_key_value_indexes=None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        **kwargs,
    ) -> tuple:
        outputs = self.model.language_model.forward_inference(
            packed_sequence=packed_query_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        return outputs


__all__ = [
    "Qwen3ForConditionalGeneration",
    "Qwen3Model",
    "Qwen3TextModel",
    "Qwen3PreTrainedModel",
    "NaiveCache",
]
