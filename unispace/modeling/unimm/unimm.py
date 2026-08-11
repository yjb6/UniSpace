# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
# flex_attention: GPU (PyTorch 2.5+) 支持，NPU 可能不支持，条件导入
try:
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    create_block_mask = None
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
)
from .qwen3_vl import NaiveCache as NaiveCache_VL
from .qwen3 import NaiveCache as NaiveCache_LM

def NaiveCache(num_layers):
    """Unified NaiveCache constructor."""
    return NaiveCache_VL(num_layers)
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm


class UnimmConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        # 新增: Qwen3VL 配置
        use_qwen_vit=True,  # 是否使用 Qwen 集成的 VIT
        qwen_model_type='qwen3vl',  # qwen2vl, qwen2.5vl, qwen3vl
        use_moe=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.use_qwen_vit = use_qwen_vit
        self.qwen_model_type = qwen_model_type
        self.use_moe = use_moe


class Unimm(PreTrainedModel):
    config_class = UnimmConfig
    base_model_prefix = 'unimm'

    def __init__(self, vl_model, vit_model, tokenizer, config: UnimmConfig):
        super().__init__(config)
        self.language_model = vl_model
        self.tokenizer = tokenizer
        self.is_qwen3vl = config.use_qwen_vit  # True=Qwen3-VL, False=Qwen3

        # Config access path differs: Qwen3-VL has nested text_config, Qwen3 is flat
        if self.is_qwen3vl:
            self.hidden_size = config.llm_config.text_config.hidden_size
            self.num_heads = config.llm_config.text_config.num_attention_heads
        else:
            self.hidden_size = config.llm_config.hidden_size
            self.num_heads = config.llm_config.num_attention_heads
        self.use_moe = config.use_moe

        # 检查 language_model 的注意力实现
        attn_impl = getattr(self.language_model.config, '_attn_implementation', 'sdpa')
        self.using_sdpa = attn_impl in ('sdpa', 'eager')
        print(f"Unimm 使用 {attn_impl} 注意力实现")

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            # Qwen3-VL: 3D MRoPE 在 RoPE 层处理 VAE 空间位置，不需要额外 pos embed
            # Qwen3 text-only: 1D RoPE 无法区分 VAE token 空间位置，需要 latent_pos_embed
            if not self.is_qwen3vl:
                self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            # 只有不使用 Qwen VIT 时才需要独立的 vit_model 和 connector
            if not config.use_qwen_vit:
                self.vit_model = vit_model
                self.vit_patch_size = config.vit_config.patch_size
                self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
                self.vit_hidden_size = config.vit_config.hidden_size
                self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
                self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
            else:
                # Qwen3VL 的 VIT 已经集成在 vl_model 中
                self.vit_model = None
                self.connector = None
                self.vit_pos_embed = None
            # 新增: Qwen3VL 支持的 patch_size
            if config.use_qwen_vit:
                self.vit_patch_size = getattr(config.vit_config, 'patch_size', 14)
                self.vit_max_num_patch_per_side = getattr(config, 'vit_max_num_patch_per_side', 70)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        # 新增: Qwen3VL 的 RoPE 计算方式
        if config.use_qwen_vit:
            from data.rope2d import get_rope_index_3, get_rope_index_25, get_rope_index_2

            if config.qwen_model_type == "qwen3vl":
                self.get_qwen_rope_index = get_rope_index_3
            elif config.qwen_model_type == "qwen2.5vl":
                self.get_qwen_rope_index = get_rope_index_25
            elif config.qwen_model_type == "qwen2vl":
                self.get_qwen_rope_index = get_rope_index_2
            else:
                raise ValueError(f"Unsupported qwen_model_type: {config.qwen_model_type}")

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.Tensor,  # 注意：可能是 3D 张量
        attention_mask: List[torch.Tensor] = None,  # 此参数被忽略
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # 新增: Qwen3VL 相关参数
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # 来自 qwen processor
        pixel_values_videos: Optional[torch.Tensor] = None,
        qwen_full_position_ids: Optional[torch.Tensor] = None,  # Qwen 计算的完整 position_ids
        # input_ids 用于 VIT placeholder 检测
        input_ids: Optional[torch.LongTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            attention_masks: 不再使用，设置为 None 即可。
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.
            # ... other args ...
        """
        embed_tokens = self.language_model.model.language_model.embed_tokens

        packed_text_embedding = embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # 根据 attention implementation 创建对应格式的掩码
        attn_impl = getattr(self.language_model.config, '_attn_implementation', 'sdpa')
        if attn_impl in ('sdpa', 'eager'):
            # SDPA / eager: 使用稠密 bool mask [1,1,S,S]
            attention_mask = create_sparse_mask(
                sample_lens, split_lens, attn_modes, packed_text_embedding.device, mask_type="sdpa"
            )
        else:
            # flex_attention: 使用 block_mask（稀疏压缩格式）
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask

        # === 视觉理解分支 ===
        if self.config.visual_und:
            if self.config.use_qwen_vit:
                # 使用 Qwen3VL 集成的 VIT 处理
                vit_extra_inputs = {}

                if pixel_values is not None:
                    vit_extra_inputs['pixel_values'] = pixel_values
                if image_grid_thw is not None:
                    vit_extra_inputs['image_grid_thw'] = image_grid_thw
                if pixel_values_videos is not None:
                    vit_extra_inputs['pixel_values_videos'] = pixel_values_videos
                if video_grid_thw is not None:
                    vit_extra_inputs['video_grid_thw'] = video_grid_thw

                # 使用 Qwen 的 position_ids
                if qwen_full_position_ids is not None:
                    packed_position_ids = qwen_full_position_ids

                # 设置图像 token 占位符的 embedding
                if packed_vit_token_indexes is not None:
                    # 获取 image_token_id (通常是 Qwen 的特殊 token，如 <|image_pad|>)
                    # 从 language_model 的 config 中获取，或者使用默认值
                    if hasattr(self.language_model, 'config') and hasattr(self.language_model.config, 'image_token_id'):
                        image_token_id = self.language_model.config.image_token_id
                    else:
                        # 默认使用一个占位符 token id (需要根据实际 tokenizer 调整)
                        image_token_id = 151655  # Qwen2VL 的 <|image_pad|> token id

                    # 创建图像占位符 token 的 embedding
                    num_image_tokens = len(packed_vit_token_indexes)
                    image_token_ids = torch.full(
                        (num_image_tokens,),
                        image_token_id,
                        dtype=torch.long,
                        device=packed_text_embedding.device
                    )
                    packed_vit_token_embed = embed_tokens(image_token_ids)

                    # 填充到 packed_sequence
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

            else:
                # 原有的独立 VIT 模型处理逻辑
                if packed_vit_tokens is not None:
                    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                    cu_seqlens = cu_seqlens.to(torch.int32)
                    max_seqlen = torch.max(vit_token_seqlens).item()
                    packed_vit_token_embed = self.vit_model(
                        packed_pixel_values=packed_vit_tokens,
                        packed_flattened_position_ids=packed_vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                    packed_vit_token_embed = self.connector(packed_vit_token_embed)
                    vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                    packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

                vit_extra_inputs = {}

        # === 视觉生成分支 ===
        if self.config.visual_gen:
            if padded_latent is not None and packed_vae_token_indexes is not None:
                p = self.latent_patch_size
                packed_latent = []
                for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                    latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                    packed_latent.append(latent)
                packed_latent_clean = torch.cat(packed_latent, dim=0)
                noise = torch.randn_like(packed_latent_clean)
                packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                packed_latent = (1 - packed_timesteps[:, None]) * noise + packed_timesteps[:, None] * packed_latent_clean
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds
                # Qwen3 text-only: 1D RoPE 无法区分 VAE 空间位置，需要 latent_pos_embed
                # Qwen3-VL: 3D MRoPE 已在 RoPE 层提供空间位置信息，无需额外 pos embed
                if not self.is_qwen3vl:
                    latent_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                    packed_latent = packed_latent + latent_pos_emb
                packed_sequence[packed_vae_token_indexes] = packed_latent
            else:
                # TimestepEmbedder FSDP dummy forward
                # 当没有 VAE 数据（纯理解 batch）时，仍需触发 FSDP all-gather
                device = packed_sequence.device
                dummy_t = torch.zeros(1, device=device)
                dummy_out = self.time_embedder(dummy_t.unsqueeze(0))
                packed_sequence = packed_sequence + dummy_out.sum() * 0

        # === 准备 language_model 的输入 ===
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if not self.config.use_qwen_vit and packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        # 如果使用 Qwen VIT (Qwen3-VL),传递视觉相关参数
        if self.is_qwen3vl and self.config.visual_und:
            extra_inputs.update(vit_extra_inputs)
            # 传递 input_ids 给 Qwen3VL 用于 get_placeholder_mask
            if input_ids is not None:
                extra_inputs['input_ids'] = input_ids

        # === 调用 language_model ===
        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        # === 损失计算 ===
        mse = None
        if self.config.visual_gen and padded_latent is not None and mse_loss_indexes is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            # target = noise - packed_latent_clean
            target = packed_latent_clean - noise
            has_mse = packed_timesteps > 0

            if has_mse.any():
                # 计算每个样本的MSE，然后取平均
                mse = (packed_mse_preds - target[has_mse]) ** 2

            else:
                mse = torch.tensor(0.0, device=packed_mse_preds.device)

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            # 使用mean reduction得到标量损失
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()
        if new_token_ids is None:
            new_token_ids = {
                'bos_token_id': tokenizer.encode("<|im_start|>")[0],
                'eos_token_id': tokenizer.encode("<|im_end|>")[0],
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                'start_of_image': tokenizer.encode("<|vision_start|>")[0],
                'end_of_image': tokenizer.encode("<|vision_end|>")[0],
            }

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # Match training format:
            #   Segment 1: [bos] user\n{caption} [eos]
            #   Segment 2: [bos] assistant\n [eos]
            user_ids = tokenizer.encode(f"user\n{prompt}")
            assistant_ids = tokenizer.encode("assistant\n")
            text_ids = (
                [new_token_ids['bos_token_id']] + user_ids + [new_token_ids['eos_token_id']]
                + [new_token_ids['bos_token_id']] + assistant_ids + [new_token_ids['eos_token_id']]
            )
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_cfg_prompts(self, curr_kvlens, curr_rope, num_images, tokenizer, new_token_ids):
        """Prepare the unconditional (CFG) text prefix.

        In training, CFG dropout only drops the ``user\\n{caption}`` segment
        but always keeps ``[bos]assistant\\n[eos]``.  The unconditional path
        must therefore encode this prefix into its KV cache.
        """
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()
        if new_token_ids is None:
            new_token_ids = {
                'bos_token_id': tokenizer.encode("<|im_start|>")[0],
                'eos_token_id': tokenizer.encode("<|im_end|>")[0],
            }

        curr = 0
        newlens, new_rope = list(), list()
        assistant_ids = tokenizer.encode("assistant\n")
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # Only the assistant prefix (matches training CFG dropout behavior)
            text_ids = (
                [new_token_ids['bos_token_id']] + assistant_ids + [new_token_ids['eos_token_id']]
            )
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.language_model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            sample_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_qwen3vit_images(self, curr_kvlens, curr_rope, images, transforms, processor, new_token_ids):
        """
        准备 VIT 图像输入，支持 Qwen3VL 处理流程
        """
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        # Qwen3VL 专用字段
        qwen_pixel_values_list = list()
        qwen_image_grid_thw_list = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()

        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # 添加 <|vision_start|> token
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # === Qwen3VL 处理分支 ===
            if self.config.use_qwen_vit:
                try:
                    from PIL import Image
                    import io

                    # 转换为 PIL Image
                    if isinstance(image, torch.Tensor):
                        img_np = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                        pil_image = Image.fromarray(img_np)
                    else:
                        pil_image = image

                    # 使用 Qwen processor 处理图像
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": "describe this image"}
                        ]
                    }]

                    qwen_result = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    )

                    # 提取 pixel_values 和 grid_thw
                    if 'pixel_values' in qwen_result:
                        qwen_pixel_values_list.append(qwen_result['pixel_values'])

                    if 'image_grid_thw' in qwen_result:
                        grid_thw = qwen_result['image_grid_thw']
                        qwen_image_grid_thw_list.append(grid_thw)

                        # 计算实际 token 数量（考虑 patch merging）
                        merge_size = getattr(self, 'qwen_merge_size', 2)
                        num_img_tokens = sum(
                            int((t * h * w) / (merge_size ** 2))
                            for t, h, w in grid_thw.tolist()
                        )
                    else:
                        # 回退逻辑
                        image_tensor = transforms(image)
                        vit_tokens = patchify(image_tensor, self.vit_patch_size)
                        num_img_tokens = vit_tokens.shape[0]
                        grid_thw = None

                    # 仍然需要保存 patchify 后的 tokens（用于索引）
                    image_tensor = transforms(image)
                    vit_patch_size = getattr(self, 'vit_patch_size', 14)  # 使用默认值
                    vit_tokens = patchify(image_tensor, vit_patch_size)
                    packed_vit_tokens.append(vit_tokens)

                    # 计算 Qwen 的 3D position_ids
                    if grid_thw is not None:
                        temp_seq_len = 1 + num_img_tokens + 1
                        temp_input_ids = torch.full(
                            (1, temp_seq_len),
                            processor.tokenizer.encode("<|image_pad|>")[0],
                            dtype=torch.long
                        )
                        temp_input_ids[0, 0] = new_token_ids['start_of_image']
                        temp_input_ids[0, -1] = new_token_ids['end_of_image']

                        # 使用 Qwen RoPE 计算方法
                        qwen_position_ids, _ = self.get_qwen_rope_index(
                            merge_size,
                            temp_input_ids,
                            image_grid_thw=grid_thw,
                            video_grid_thw=None,
                        )

                        # 提取图像 token 的 position_ids [3, num_img_tokens]
                        image_position_ids = qwen_position_ids.squeeze(1)[:, 1:1+num_img_tokens]
                        packed_vit_position_ids.append(image_position_ids)
                    else:
                        # 回退到原始逻辑
                        vit_position_ids = self.get_flattened_position_ids(
                            image_tensor.size(1), image_tensor.size(2),
                            self.vit_patch_size,
                            max_num_patches_per_side=self.vit_max_num_patch_per_side
                        )
                        packed_vit_position_ids.append(vit_position_ids)

                except Exception as e:
                    print(f"警告: Qwen 图像处理失败，回退到原始逻辑: {e}")
                    # 完全回退到原始逻辑
                    image_tensor = transforms(image)
                    vit_tokens = patchify(image_tensor, self.vit_patch_size)
                    num_img_tokens = vit_tokens.shape[0]
                    packed_vit_tokens.append(vit_tokens)

                    vit_position_ids = self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.vit_patch_size,
                        max_num_patches_per_side=self.vit_max_num_patch_per_side
                    )
                    packed_vit_position_ids.append(vit_position_ids)

            # === 原始处理逻辑 ===
            else:
                image_tensor = transforms(image)
                vit_tokens = patchify(image_tensor, self.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                packed_vit_tokens.append(vit_tokens)

                vit_position_ids = self.get_flattened_position_ids(
                    image_tensor.size(1), image_tensor.size(2),
                    self.vit_patch_size,
                    max_num_patches_per_side=self.vit_max_num_patch_per_side
                )
                packed_vit_position_ids.append(vit_position_ids)

            # 记录 token 索引
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            # 添加 <|vision_end|> token
            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # 更新 position_ids
            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        # 构建返回字典
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        # 处理 Qwen 的 position_ids 格式
        if len(packed_vit_position_ids) > 0:
            if packed_vit_position_ids[0].dim() == 2:
                # Qwen 3D position_ids: [3, num_tokens]
                packed_vit_position_ids_tensor = torch.cat(packed_vit_position_ids, dim=1)

                # 构建完整的 3D position_ids
                total_seq_len = _curr
                qwen_full_position_ids = torch.zeros(3, total_seq_len, dtype=torch.long)

                # 非图像 token 使用原始 position_ids
                for i in range(3):
                    qwen_full_position_ids[i] = generation_input['packed_position_ids']

                # 图像 token 使用 Qwen 的 3D position_ids
                vit_indexes = generation_input['packed_vit_token_indexes']
                qwen_full_position_ids[:, vit_indexes] = packed_vit_position_ids_tensor

                generation_input['packed_position_ids'] = qwen_full_position_ids
                generation_input['packed_vit_position_ids'] = packed_vit_position_ids_tensor
            else:
                # 原始 1D position_ids
                generation_input['packed_vit_position_ids'] = torch.cat(packed_vit_position_ids, dim=0)

        # 添加 Qwen 专用字段
        if len(qwen_pixel_values_list) > 0:
            generation_input['pixel_values'] = torch.cat(qwen_pixel_values_list, dim=0)
        if len(qwen_image_grid_thw_list) > 0:
            generation_input['image_grid_thw'] = torch.cat(qwen_image_grid_thw_list, dim=0)

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        embed_tokens = self.language_model.model.language_model.embed_tokens

        packed_text_embedding = embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # === 视觉理解分支 ===
        if self.config.visual_und:
            # === Qwen3VL 分支 ===
            if self.config.use_qwen_vit:
                # 构建 Qwen VIT 的额外参数
                vit_extra_inputs = {}
                if pixel_values is not None:
                    vit_extra_inputs['pixel_values'] = pixel_values
                if image_grid_thw is not None:
                    vit_extra_inputs['image_grid_thw'] = image_grid_thw
                if video_grid_thw is not None:
                    vit_extra_inputs['video_grid_thw'] = video_grid_thw

                # 使用 Qwen 的 position_ids（如果是 3D 格式）
                if packed_position_ids.dim() == 2 and packed_position_ids.shape[0] == 3:
                    # Qwen 3D position_ids: [3, seq_len]
                    qwen_full_position_ids = packed_position_ids
                else:
                    qwen_full_position_ids = None

                # 设置图像 token 占位符的 embedding
                if packed_vit_token_indexes is not None:
                    # 获取 image_token_id
                    if hasattr(self.language_model, 'config') and hasattr(self.language_model.config, 'image_token_id'):
                        image_token_id = self.language_model.config.image_token_id
                    else:
                        image_token_id = 151655  # Qwen 默认的 <|image_pad|> token id
                    # 创建图像占位符 token 的 embedding
                    num_image_tokens = len(packed_vit_token_indexes)
                    image_token_ids = torch.full(
                        (num_image_tokens,),
                        image_token_id,
                        dtype=torch.long,
                        device=packed_text_embedding.device
                    )
                    packed_vit_token_embed = embed_tokens(image_token_ids)
                    # 填充到 packed_sequence
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
            # === 原始独立 VIT 模型分支 ===
            else:
                if packed_vit_tokens is not None:
                    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                    cu_seqlens = cu_seqlens.to(torch.int32)
                    max_seqlen = torch.max(vit_token_seqlens).item()
                    # 使用独立的 VIT 模型处理
                    packed_vit_token_embed = self.vit_model(
                        packed_pixel_values=packed_vit_tokens,
                        packed_flattened_position_ids=packed_vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                    packed_vit_token_embed = self.connector(packed_vit_token_embed)
                    vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                    packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                    if packed_vit_token_embed.dtype != packed_sequence.dtype:
                        packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
                vit_extra_inputs = {}

        # === 准备 extra_inputs ===
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        # 如果使用 Qwen VIT，传递视觉相关参数
        if self.config.use_qwen_vit and self.config.visual_und:
            extra_inputs.update(vit_extra_inputs)
            # 使用 Qwen 的 3D position_ids（如果存在）
            if qwen_full_position_ids is not None:
                packed_position_ids = qwen_full_position_ids


        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            if self.is_qwen3vl:
                # 3D MRoPE position_ids for VAE tokens
                st_idx = curr_position_id
                t_pos = torch.full((num_img_tokens + 2,), st_idx, dtype=torch.long)
                h_grid, w_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                h_pos = torch.cat([torch.tensor([st_idx]), h_grid.flatten() + st_idx, torch.tensor([st_idx + h - 1])])
                w_pos = torch.cat([torch.tensor([st_idx]), w_grid.flatten() + st_idx, torch.tensor([st_idx + w - 1])])
                vae_pos_3d = torch.stack([t_pos, h_pos, w_pos], dim=0)  # [3, N+2]
                packed_position_ids.append(vae_pos_3d)
            else:
                # 1D position_ids: 所有 VAE tokens 共享同一个 RoPE 位置 st_idx
                # (与训练一致：训练时 3D position_ids dim 0 = 常数 st_idx，
                #  Qwen3TextModel.forward_train 取 dim[0]，空间位置靠 latent_pos_embed)
                st_idx = curr_position_id
                pos_1d = torch.full((num_img_tokens + 2,), st_idx, dtype=torch.long)
                packed_position_ids.append(pos_1d)

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            # new_rope: 与训练数据管线 dataset_base.py line 610 一致
            new_rope.append(max(st_idx + h, st_idx + w) + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        if self.is_qwen3vl:
            # 组装 3D position_ids [3, total_seq_len]
            total_seq_len = _curr
            full_position_ids_3d = torch.zeros(3, total_seq_len, dtype=torch.long)
            pos_offset = 0
            for pos_3d in packed_position_ids:
                seg_len = pos_3d.shape[1]
                full_position_ids_3d[:, pos_offset:pos_offset + seg_len] = pos_3d
                pos_offset += seg_len
            final_position_ids = full_position_ids_3d  # [3, seq_len]
        else:
            # 组装 1D position_ids [seq_len]
            final_position_ids = torch.cat(packed_position_ids, dim=0)

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": final_position_ids,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
    ):
        packed_text_embedding = self.language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        # 与训练 forward() 保持一致的顺序: vae2llm → timestep_embeds → latent_pos_embed
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds
        # Qwen3 text-only: 1D RoPE 无法区分 VAE 空间位置，需要 latent_pos_embed
        # Qwen3-VL: 3D MRoPE 已在 RoPE 层提供空间位置信息，无需额外 pos embed
        if not self.is_qwen3vl:
            latent_pos_emb = self.latent_pos_embed(packed_vae_position_ids)
            packed_latent = packed_latent + latent_pos_emb
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, generator=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(
                    num_image_tokens, self.latent_channel * self.latent_patch_size ** 2,
                    generator=generator,
                )
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if self.is_qwen3vl:
                # 3D MRoPE position_ids for VAE tokens
                st_idx = curr_position_id
                t_pos = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                h_grid, w_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                h_pos = torch.cat([torch.tensor([st_idx]), h_grid.flatten() + st_idx, torch.tensor([st_idx + h - 1])])
                w_pos = torch.cat([torch.tensor([st_idx]), w_grid.flatten() + st_idx, torch.tensor([st_idx + w - 1])])
                vae_pos_3d = torch.stack([t_pos, h_pos, w_pos], dim=0)  # [3, N+2]
                packed_position_ids.append(vae_pos_3d)
            else:
                # 1D position_ids: 所有 VAE tokens 共享同一个 RoPE 位置 st_idx
                # (与训练一致：训练时 3D position_ids dim 0 = 常数 st_idx)
                st_idx = curr_position_id
                pos_1d = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                packed_position_ids.append(pos_1d)
            packed_seqlens.append(num_image_tokens + 2)

        if self.is_qwen3vl:
            # 组装 3D position_ids [3, total_seq_len]
            total_seq_len = query_curr
            full_position_ids_3d = torch.zeros(3, total_seq_len, dtype=torch.long)
            pos_offset = 0
            for pos_3d in packed_position_ids:
                seg_len = pos_3d.shape[1]
                full_position_ids_3d[:, pos_offset:pos_offset + seg_len] = pos_3d
                pos_offset += seg_len
            final_position_ids = full_position_ids_3d  # [3, seq_len]
        else:
            # 组装 1D position_ids [seq_len]
            final_position_ids = torch.cat(packed_position_ids, dim=0)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": final_position_ids,
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if self.is_qwen3vl:
                # 3D MRoPE position_ids for VAE tokens
                st_idx = curr_position_id
                t_pos = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                h_grid, w_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                h_pos = torch.cat([torch.tensor([st_idx]), h_grid.flatten() + st_idx, torch.tensor([st_idx + h - 1])])
                w_pos = torch.cat([torch.tensor([st_idx]), w_grid.flatten() + st_idx, torch.tensor([st_idx + w - 1])])
                vae_pos_3d = torch.stack([t_pos, h_pos, w_pos], dim=0)  # [3, N+2]
                packed_position_ids.append(vae_pos_3d)
            else:
                # 1D position_ids: 所有 VAE tokens 共享同一个 RoPE 位置 st_idx
                # (与训练一致：训练时 3D position_ids dim 0 = 常数 st_idx)
                st_idx = curr_position_id
                pos_1d = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                packed_position_ids.append(pos_1d)

        if self.is_qwen3vl:
            # 组装 3D position_ids [3, total_seq_len]
            total_seq_len = query_curr
            full_position_ids_3d = torch.zeros(3, total_seq_len, dtype=torch.long)
            pos_offset = 0
            for pos_3d in packed_position_ids:
                seg_len = pos_3d.shape[1]
                full_position_ids_3d[:, pos_offset:pos_offset + seg_len] = pos_3d
                pos_offset += seg_len
            final_position_ids = full_position_ids_3d  # [3, seq_len]
        else:
            # 组装 1D position_ids [seq_len]
            final_position_ids = torch.cat(packed_position_ids, dim=0)

        generation_input = {
            "cfg_packed_position_ids": final_position_ids,
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        generation_input_text=None,
        inference_mode="flex",
        max_num_tokens: int = 9216,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        # if enable_taylorseer:
        #     self.language_model.enable_taylorseer = True
        #     model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
        #     model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
        #     model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        # else:
        #     self.language_model.enable_taylorseer = False
            # model_pred_cache_dic, model_pred_current = None, None
            # model_pred_text_cache_dic, model_pred_text_current = None, None
            # model_pred_img_cache_dic, model_pred_img_current = None, None

        model_pred_cache_dic, model_pred_current = None, None
        model_pred_text_cache_dic, model_pred_text_current = None, None
        model_pred_img_cache_dic, model_pred_img_current = None, None

        x_t = packed_init_noises

        if inference_mode == "flex":
            # 检查是否使用 SDPA
            attn_impl = getattr(self.language_model.config, '_attn_implementation', 'sdpa')
            if attn_impl in ('sdpa', 'eager'):
                # SDPA 需要标准的注意力掩码
                device = x_t.device
                total_seq_len = max_num_tokens

                # 创建因果注意力掩码
                attention_mask = torch.full(
                    (1, 1, total_seq_len, total_seq_len),
                    float('-inf'),
                    device=device
                )
                causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len, device=device))
                attention_mask = attention_mask.masked_fill(causal_mask.bool(), 0.0)

                # CFG 掩码
                cfg_attention_mask = attention_mask.clone()
            else:
                # 原有的 flex_attention block_mask 创建逻辑
                device = x_t.device
                key_values_lens_val = key_values_lens.item() if key_values_lens.numel() == 1 else key_values_lens[0].item()
                packed_seqlens_val = packed_seqlens.item() if packed_seqlens.numel() == 1 else packed_seqlens[0].item()

                sample_lens = []
                sample_lens.append(key_values_lens_val + packed_seqlens_val)
                sample_lens.append(max_num_tokens - sum(sample_lens))

                split_lens = [key_values_lens_val, packed_seqlens_val, sample_lens[-1]]
                attn_modes = ['casual'] * len(split_lens)
                attn_modes[1] = 'full'

                cfg_sample_lens = [packed_seqlens_val]
                cfg_sample_lens.append(max_num_tokens - sum(cfg_sample_lens))
                cfg_split_lens = [packed_seqlens_val, cfg_sample_lens[-1]]
                cfg_attn_modes = ['full', 'casual']

                sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
                cfg_sparse_mask = create_sparse_mask(cfg_sample_lens, cfg_split_lens, cfg_attn_modes, device)

                total_seqlen = sum(sample_lens)
                total_cfg_seqlen = sum(cfg_sample_lens)

                block_mask = create_block_mask(
                    sparse_mask,
                    B=1,
                    H=self.num_heads,
                    Q_LEN=total_seqlen,
                    KV_LEN=total_seqlen,
                    device=device,
                    BLOCK_SIZE=128,
                    _compile=True
                )

                cfg_block_mask = create_block_mask(
                    cfg_sparse_mask,
                    B=1,
                    H=self.num_heads,
                    Q_LEN=total_cfg_seqlen,
                    KV_LEN=total_cfg_seqlen,
                    device=device,
                    BLOCK_SIZE=128,
                    _compile=True
                )
        else:
            block_mask = None
            cfg_block_mask = None
            attention_mask = None
            cfg_attention_mask = None
            prompt_token_lens = 0

        timesteps = torch.linspace(0, 1, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        # timesteps = timesteps[:-1]
        timesteps = timesteps[1:]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                attention_mask=block_mask,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_attention_mask=cfg_block_mask,
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        # if enable_taylorseer:
        #     del model_pred_cache_dic, model_pred_current
        #     del model_pred_text_cache_dic, model_pred_text_current
        #     del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - prompt_token_lens - 2).tolist())
        return unpacked_latent

    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        attention_mask=None,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_attention_mask=None,
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        # Qwen3 text-only: 1D RoPE 无法区分 VAE 空间位置，需要 latent_pos_embed
        # Qwen3-VL: 3D MRoPE 已在 RoPE 层提供空间位置信息，无需额外 pos embed
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds
        if not self.is_qwen3vl:
            latent_pos_emb = self.latent_pos_embed(packed_vae_position_ids)
            x_t = x_t + latent_pos_emb
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        # 调用 language_model 时传递正确的注意力掩码
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            sample_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            attention_mask=attention_mask,  # 传递 SDPA 格式的掩码
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.last_hidden_state)
        v_t = v_t[packed_vae_token_indexes]

        # CFG 分支
        if cfg_text_scale > 1.0:
            cfg_packed_sequence = packed_sequence[packed_vae_token_indexes[0] - 1:]
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=cfg_packed_sequence,
                sample_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                attention_mask=cfg_attention_mask,  # 传递 CFG 掩码
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.last_hidden_state)
            cfg_packed_vae_token_indexes = packed_vae_token_indexes - (packed_vae_token_indexes[0] - 1)
            cfg_text_v_t = cfg_text_v_t[cfg_packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.last_hidden_state)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.language_model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.last_hidden_state
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

