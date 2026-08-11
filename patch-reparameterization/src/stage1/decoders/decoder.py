# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViT MAE (masked autoencoder) model."""

import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union, Dict

import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint # {{ edit_1 }} 添加 checkpoint 模块

# correct the above import to the following
from .utils import ViTMAEConfig, ACT2FN, ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from stage1.encoders.siglip2 import build_adapter
import logging
logger = logging.getLogger(__name__)

@dataclass
class ViTMAEModelOutput(ModelOutput):
    """
    Class for ViTMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """
    Class for ViTMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cls_classifier_output: Optional[torch.FloatTensor] = None


@dataclass
class ViTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, num_positions):
    """
    Create 1D sin/cos positional embeddings for side tokens.

    Args:
        embed_dim: Embedding dimension (must be even)
        num_positions: Number of positions to encode

    Returns:
        (num_positions, embed_dim): Position embeddings
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    pos = np.arange(num_positions, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, 0, :]
        patch_pos_embed = self.position_embeddings[:, 1:, :]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(h0) != patch_pos_embed.shape[-2] or int(w0) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTMAE
class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSdpaSelfAttention ViT->ViTMAE
class ViTMAESdpaSelfAttention(ViTMAESelfAttention):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTMAE
class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTMAE
class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)


        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTMAE
class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->ViTMAE
class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states



# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTMAE,VIT->VITMAE
class ViTMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config) # no SPDA by default
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class GeneralDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        # self.gradient_checkpointing = False
        self.gradient_checkpointing = getattr(config, "use_checkpoint", False)
        logger.info(f"gradient_checkpointing: {self.gradient_checkpointing}")
        self.config = config
        self.num_patches = num_patches
        self.initialize_weights(num_patches)
        self.decoder_config = decoder_config

        # Side token projection: 根据配置决定是否创建
        # 从encoder的hidden_size投影到decoder的hidden_size
        use_side_tokens = getattr(config, "use_side_tokens", False)
        self.use_side_tokens = use_side_tokens

        if use_side_tokens:
            logger.info("use_side_tokens: %s", use_side_tokens)
            self.side_token_proj = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

            # 根据配置决定 side tokens 的 position embedding 类型
            # "learnable": 可学习的 position embeddings
            # "sin_cos_1d": 1D sin-cos position embeddings (固定，不可学习)
            # "sin_cos_2d": 2D sin-cos position embeddings (固定，不可学习，需要 grid shape)
            # "none": 不使用 position embeddings (全0)
            self.side_token_pos_embed_type = getattr(config, "side_token_pos_embed_type", "none")
            num_side_tokens = getattr(config, "side_tokens_num", None)
            if num_side_tokens is None:
                raise ValueError("side_tokens_num must be provided when using side tokens")
            self.side_token_pos_embed = nn.Parameter(
                    torch.zeros(1, num_side_tokens, config.decoder_hidden_size)
                )
            if self.side_token_pos_embed_type == "learnable":
                # 可学习的 position embeddings
                side_tokens_num = getattr(config, "side_tokens_num", None)
                if side_tokens_num is None:
                    raise ValueError("side_tokens_num must be provided when using learnable position embeddings")

                nn.init.normal_(self.side_token_pos_embed, std=0.02)
            elif self.side_token_pos_embed_type in ["sin_cos_1d", "sin_cos_2d"]:
                # sin-cos position embeddings (固定，不可学习)
                if num_side_tokens is None:
                    raise ValueError("num_side_tokens must be provided when using sin_cos position embeddings")

                self.side_token_pos_embed = nn.Parameter(
                    torch.zeros(1, num_side_tokens, config.decoder_hidden_size),
                    requires_grad=False  # 固定，不可学习
                )
                # 在 init_side_weights 中初始化
                self.init_side_weights()
            else:
                #零初始化
                nn.init.zeros_(self.side_token_pos_embed)
                logger.info("zero initialized side token position embeddings")

        else:
            logger.info("decoder not use side tokens")
            self.side_token_proj = None

        use_cls_classifier = getattr(config, "use_cls_classifier", False)
        self.use_cls_classifier = use_cls_classifier
        if use_cls_classifier:
            logger.info("use cls classifier")
            self.cls_classifier = nn.Sequential(
                nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(config.decoder_hidden_size, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.cls_classifier[2].weight)
            nn.init.zeros_(self.cls_classifier[2].bias)
        else:
            logger.info("not use cls classifier")
            self.cls_classifier = None

        self.set_trainable_cls_token()





        # 加载预训练decoder权重（在初始化权重之后）
        self._load_pretrained_weights(config)

    def set_trainable_cls_token(self, tensor: Optional[torch.Tensor] = None):
        # register a trainable CLS token
        tensor = torch.zeros(1, 1, self.decoder_config.hidden_size) if tensor is None else tensor
        self.trainable_cls_token = nn.Parameter(tensor)

    def set_training_mode(
        self,
        mode: str = "full",
        return_stats: bool = False,
        last_n_layers: Optional[int] = None,
    ):
        """
        设置 decoder 哪些参数可学，与 encoder 的 set_training_mode 用法一致。
        mode:
            - "full": 全部可学
            - "decoder_pred_only": 只训练 decoder_pred（输出头）
            - "decoder_pred_and_norm": decoder_norm + decoder_pred
            - "last_n_layers": 只训练最后 last_n_layers 层 + decoder_norm + decoder_pred（需传 last_n_layers=N）
        """
        self.requires_grad_(False)
        if mode == "full":
            logger.info("GeneralDecoder: training full model")
            self.requires_grad_(True)
        elif mode == "decoder_pred_only":
            logger.info("GeneralDecoder: training decoder_pred only")
            for p in self.decoder_pred.parameters():
                p.requires_grad = True
        elif mode == "decoder_pred_and_norm":
            logger.info("GeneralDecoder: training decoder_norm + decoder_pred")
            for m in (self.decoder_norm, self.decoder_pred):
                for p in m.parameters():
                    p.requires_grad = True
        elif mode == "last_n_layers":
            if last_n_layers is None:
                raise ValueError("last_n_layers mode requires last_n_layers=N")
            n = int(last_n_layers)
            logger.info("GeneralDecoder: training last %d layers + decoder_norm + decoder_pred", n)
            for p in self.decoder_norm.parameters():
                p.requires_grad = True
            for p in self.decoder_pred.parameters():
                p.requires_grad = True
            for layer in self.decoder_layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unknown decoder training mode: {mode}")

    def _load_pretrained_weights(self, config):
        """从config中读取预训练权重路径并加载"""
        # 支持两种方式：
        # 1. pretrained_decoder_path: 纯decoder state_dict路径
        # 2. pretrained_decoder_checkpoint + pretrained_decoder_use_ema: 完整checkpoint路径

        pretrained_path = getattr(config, "pretrained_decoder_path", None)
        pretrained_checkpoint = getattr(config, "pretrained_decoder_checkpoint", None)
        use_ema = getattr(config, "pretrained_decoder_use_ema", True)

        if not (pretrained_path or pretrained_checkpoint):
            return  # 没有指定预训练权重，跳过

        if pretrained_path and pretrained_checkpoint:
            raise ValueError("Cannot specify both pretrained_decoder_path and pretrained_decoder_checkpoint. Choose one.")

        load_path = pretrained_checkpoint if pretrained_checkpoint else pretrained_path

        try:
            checkpoint = torch.load(load_path, map_location="cpu")

            # 判断是完整checkpoint还是纯decoder state_dict
            if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
                # 完整checkpoint格式，提取decoder部分
                if use_ema:
                    if "ema" not in checkpoint:
                        raise KeyError(f"Checkpoint has no 'ema' key. Set pretrained_decoder_use_ema=False or use a different checkpoint.")
                    state_dict = checkpoint["ema"]
                else:
                    if "model" not in checkpoint:
                        raise KeyError(f"Checkpoint has no 'model' key. Set pretrained_decoder_use_ema=True or use a different checkpoint.")
                    state_dict = checkpoint["model"]

                # 提取decoder部分的权重（处理各种包装前缀）
                decoder_state_dict = {}
                for key, value in state_dict.items():
                    # 处理可能的包装前缀：module., _orig_mod., module._orig_mod.
                    clean_key = key
                    for prefix in ["module._orig_mod.decoder.", "_orig_mod.decoder.", "module.decoder.", "decoder."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break

                    # 只保留decoder相关的键
                    if clean_key != key or key.startswith("decoder."):
                        decoder_state_dict[clean_key] = value

                if len(decoder_state_dict) == 0:
                    raise ValueError(f"No decoder weights found in checkpoint. Check key prefixes.")
            else:
                # 假设是纯decoder state_dict
                decoder_state_dict = checkpoint

            # Filter out keys with shape mismatch (e.g. decoder_embed when encoder hidden_size differs)
            current_state = self.state_dict()
            filtered_state_dict = {}
            shape_mismatch_keys = []
            for k, v in decoder_state_dict.items():
                if k in current_state and v.shape != current_state[k].shape:
                    shape_mismatch_keys.append(f"{k}: ckpt {v.shape} vs model {current_state[k].shape}")
                else:
                    filtered_state_dict[k] = v
            if shape_mismatch_keys:
                logger.warning(f"[GeneralDecoder] Skipping shape-mismatched keys: {shape_mismatch_keys}")

            # 加载权重
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                logger.warning(f"[GeneralDecoder] Warning: Missing keys when loading pretrained decoder: {missing_keys} ")
            if unexpected_keys:
                logger.warning(f"[GeneralDecoder] Warning: Unexpected keys when loading pretrained decoder: {unexpected_keys} ")

            logger.info(f"[GeneralDecoder] Loaded pretrained decoder weights from {load_path}")
                    # 从 decoder_embed 初始化 side_token_proj（如果启用了 side tokens 且权重未加载）
            if self.use_side_tokens and self.side_token_proj is not None:
                # 检查 side_token_proj 是否在 missing_keys 中（说明预训练权重中没有）
                side_proj_missing = any('side_token_proj' in k for k in missing_keys)
                if side_proj_missing:
                    self.side_token_proj.weight.data.copy_(self.decoder_embed.weight.data)
                    if self.side_token_proj.bias is not None and self.decoder_embed.bias is not None:
                        self.side_token_proj.bias.data.copy_(self.decoder_embed.bias.data)
                    logger.info("[GeneralDecoder] Initialized side_token_proj from decoder_embed")
                else:
                    logger.info("[GeneralDecoder] side_token_proj loaded from pretrained weights")

        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained decoder from {load_path}: {e}")

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        This method is a modified version of the interpolation function for ViT-mae model at the deocder, that
        allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # -1 removes the class dimension since we later append it without interpolation
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        # Separation of class token and patch tokens
        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]

        # To retain the final 3d tensor with the required dimensions
        dim = self.decoder_pos_embed.shape[-1]

        # Increasing a dimension to enable bicubic interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)

        # permute to bring the dimension to be interpolated, to the last
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolating the decoder position embeddings shape wrt embeddings shape i.e (x).
        # 1 keeps the other dimension constant
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )

        # Converting back to the original shape
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # Adding the class token back
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        # interpolate the latent
        #print(f"interpolating latent from {l} to {self.num_patches}, x.shape = {x.shape}")
        h, w = int(l**0.5), int(l**0.5)
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        target_size = (int(self.num_patches**0.5), int(self.num_patches**0.5))
        x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def init_side_weights(self):
        """
        Initialize side tokens position embeddings based on configuration.
        """
        if not hasattr(self, 'side_token_pos_embed') or self.side_token_pos_embed is None:
            return

        if self.side_token_pos_embed_type == "sin_cos_2d":
            # 2D sin-cos position embeddings (默认正方形)
            num_side_tokens = self.side_token_pos_embed.shape[1]
            grid_size = int(np.sqrt(num_side_tokens))
            # 确保是正方形
            if grid_size * grid_size != num_side_tokens:
                raise ValueError(
                    f"num_side_tokens ({num_side_tokens}) must be a perfect square for sin_cos_2d"
                )
            side_pos_embed = get_2d_sincos_pos_embed(
                self.side_token_pos_embed.shape[-1],
                grid_size,
                add_cls_token=False
            )
            self.side_token_pos_embed.data.copy_(
                torch.from_numpy(side_pos_embed).float().unsqueeze(0)
            )
        elif self.side_token_pos_embed_type == "sin_cos_1d":
            # 1D sin-cos position embeddings
            num_side_tokens = self.side_token_pos_embed.shape[1]
            side_pos_embed = get_1d_sincos_pos_embed(
                self.side_token_pos_embed.shape[-1],
                num_side_tokens
            )
            self.side_token_pos_embed.data.copy_(
                torch.from_numpy(side_pos_embed).float().unsqueeze(0)
            )
        # learnable 类型已经在 __init__ 中用 normal_ 初始化了，不需要在这里处理


    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`Tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def set_gradient_checkpointing(self):
        if not self.gradient_checkpointing:
            logger.info("Setting gradient checkpointing to True")
            self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
        side_tokens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        debug: bool = False
    ):
        # embed tokens
        # logger.debug(f"hidden_states shape: {hidden_states.shape}")
        x = self.decoder_embed(hidden_states)
        #print(f"x.shape = {x.shape}")
        if drop_cls_token:
            x_ = x[:, 1:, :]  # no cls token
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x) # interpolate the whole latent

        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)

        # Process side tokens if provided
        side_tokens_processed = None
        num_side_tokens = 0



        if side_tokens is not None:
            # Handle different input formats:
            # 1. Dict format: {group_name: tensor} - 合并所有groups
            # 2. Tensor format: (batch_size, num_side_tokens, hidden_size)
            if isinstance(side_tokens, dict):
                # 合并所有side token groups
                # 假设每个group的tensor形状是 (batch_size * num_queries, hidden_size)
                # 需要reshape回 (batch_size, num_queries, hidden_size) 然后拼接
                batch_size = x_.shape[0]
                side_token_list = []
                for group_name, group_tokens in side_tokens.items():
                    # group_tokens shape: (batch_size * num_queries, hidden_size)
                    # 需要reshape为 (batch_size, num_queries, hidden_size)
                    if group_tokens.shape[0] % batch_size != 0:
                        raise ValueError(
                            f"Side tokens group '{group_name}' shape {group_tokens.shape[0]} "
                            f"must be divisible by batch_size {batch_size}"
                        )
                    num_queries = group_tokens.shape[0] // batch_size
                    group_tokens_reshaped = group_tokens.view(batch_size, num_queries, -1)
                    side_token_list.append(group_tokens_reshaped)
                # 拼接所有groups: (batch_size, total_side_tokens, hidden_size)
                if side_token_list:
                    side_tokens_tensor = torch.cat(side_token_list, dim=1)
                else:
                    side_tokens_tensor = None
            else:
                # Tensor format: (batch_size, num_side_tokens, hidden_size)
                side_tokens_tensor = side_tokens

            if side_tokens_tensor is not None:
                # if debug:
                #     logger.warning("debug info is enabled, set side_tokens_tensor to zeros")
                #     side_tokens_tensor = torch.zeros_like(side_tokens_tensor)
                # 投影side tokens到decoder维度
                # side_tokens_tensor: (batch_size, num_side_tokens, encoder_hidden_size)
                # -> (batch_size, num_side_tokens, decoder_hidden_size)
                side_tokens_processed = self.side_token_proj(side_tokens_tensor)
                num_side_tokens = side_tokens_processed.shape[1]
                # side tokens不需要pos encoding（按用户要求）
        else:
            if self.use_side_tokens:
                logger.warning("use_side_tokens is True, but side_tokens is None")

        # 拼接tokens: [CLS, side_tokens (optional), patch_tokens]
        if side_tokens_processed is not None:
            x = torch.cat([cls_token, side_tokens_processed, x_], dim=1)
        else:
            x = torch.cat([cls_token, x_], dim=1)

        # add pos embed
        if interpolate_pos_encoding:
            assert drop_cls_token, "interpolate_pos_encoding only works with drop_cls_token=True"
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        # 应用pos encoding:
        # - CLS token: 使用 pos_embed[:, 0, :]
        # - Side tokens: 不加pos encoding（全0）
        # - Patch tokens: 使用 pos_embed[:, 1:, :]
        cls_pos_embed = decoder_pos_embed[:, 0:1, :]  # (1, 1, decoder_hidden_size)
        patch_pos_embed = decoder_pos_embed[:, 1:, :]  # (1, num_patches, decoder_hidden_size)

        # 构建完整的pos embedding
        if num_side_tokens > 0:
            # CLS + side tokens (no pos) + patch tokens
            side_pos_embed = self.side_token_pos_embed
            full_pos_embed = torch.cat([cls_pos_embed, side_pos_embed, patch_pos_embed], dim=1)
        else:
            # CLS + patch tokens
            full_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)

        hidden_states = x + full_pos_embed
        #print(f"hidden_states.shape = {hidden_states.shape}")
        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None


        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         layer_module.__call__,
            #         hidden_states,
            #         None,
            #         output_attentions,
            #     )
            if self.gradient_checkpointing and self.training:
                # 使用更简洁的 lambda 或局部函数，并显式处理 use_reentrant
                def create_custom_forward(module):
                    def custom_forward(x, mask):
                        return module(x, head_mask=mask, output_attentions=output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,  # head_mask
                    use_reentrant=False # 建议设置为 False 以获得更好的兼容性
                )
            else:

                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.use_cls_classifier:
            cls_hidden_states = hidden_states[:, 0:1, :]
            cls_classifier_output = self.cls_classifier(cls_hidden_states)
        else:
            cls_classifier_output = None
        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token and side tokens (if any)
        # 移除 CLS token (索引0) 和 side tokens (如果有的话)
        if num_side_tokens > 0:
            logits = logits[:, 1 + num_side_tokens:, :]  # 跳过 CLS 和 side tokens
        else:
            logits = logits[:, 1:, :]  # 只移除 CLS token
        # logger.debug(f"logits shape: {logits.shape}")
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cls_classifier_output=cls_classifier_output,
        )


