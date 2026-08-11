# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# MOT (Mixture-of-Transformers) version of Unimm model.
# Uses Qwen3VLMoTForConditionalGeneration instead of Qwen3VLForConditionalGeneration.

import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from .qwen3_mot import _scatter_at
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
from .qwen3_vl import NaiveCache as _NaiveCacheVL
from .qwen3 import NaiveCache as _NaiveCacheLM
NaiveCache = _NaiveCacheVL  # default, both implementations are identical
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding


class ExactMergerConnector(nn.Module):
    """
    unified2llm_und connector that replicates the official Qwen3VL PatchMerger behavior
    on RAE merged tokens (4 patches × [sem=1152, recon=128] = 5120 dims per token).

    Weights are initialized from the official Qwen3VL merger and can be fine-tuned.
    Architecture matches _ExactMergerConnector in the adapter (norm + fc1 + fc2).
    """
    def __init__(self, total_in: int, merger_hidden: int, out_dim: int, sem_per_patch: int):
        super().__init__()
        self.total_in      = total_in       # 5120  (4 × z_per_patch)
        self.merger_hidden = merger_hidden  # 4608  (4 × sem_per_patch)
        self.out_dim       = out_dim        # 4096  (LLM hidden size)
        self.sem           = sem_per_patch  # 1152
        self.z             = total_in // 4  # 1280  (z_per_patch = sem + recon)

        self.norm = nn.LayerNorm(sem_per_patch, eps=1e-6)
        self.fc1  = nn.Linear(total_in, merger_hidden, bias=True)
        self.fc2  = nn.Linear(merger_hidden, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, total_in=5120]
        N = x.shape[0]
        x4 = x.reshape(N, 4, self.z)                                        # [N, 4, 1280]
        x_sem_n = self.norm(x4[:, :, :self.sem])                            # [N, 4, 1152]
        x_fc1 = torch.cat([x_sem_n, x4[:, :, self.sem:]], dim=-1)           # [N, 4, 1280]
        x_fc1 = x_fc1.reshape(N, self.total_in)                             # [N, 5120]
        return self.fc2(torch.nn.functional.gelu(self.fc1(x_fc1)))

from tqdm import tqdm

# ── mark_dynamic for MOT index routing (torch.compile compatible) ──
# The MOT decoder layer uses variable-length index tensors (und_indexes, gen_indexes)
# for gather/scatter/project routing. We use torch._dynamo.mark_dynamic() to mark
# only index tensor dim=0 as symbolic, so dynamo generates ONE compiled graph that
# handles all index lengths — zero recompilation, zero padding, zero compute waste.
#
# Why mark_dynamic instead of torch.compile(dynamic=True)?
#   dynamic=True makes ALL tensor dims symbolic, including flex_attention Q/K/V.
#   PT 2.5.1 inductor cannot generate Triton kernels for flex_attention with
#   symbolic sequence dims (NoValidChoicesError). mark_dynamic is selective.
#
# BlockMask (for flex_attention) has FIXED shape (1, H, seq_len, seq_len) — static.
# Only the index tensors get symbolic dim=0.
#
# Key: None values must be avoided (dynamo compiles different graphs for None vs Tensor).
# When one side (und or gen) is empty, pass a length-1 sink tensor instead of None.

_SINK_LEN = 1  # minimal index tensor length for empty und/gen


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
        # 新增: Qwen3Unified 统一表征配置
        use_qwen3_unified=False,  # 理解+生成共用同一个 Qwen3Unified encoder
        use_spatial_merge=False,  # 2x2 spatial merge after encoder (4x token reduction)
        use_spatial_merge_gen=None,  # gen 侧 merge（None=跟随 use_spatial_merge）
        use_spatial_merge_und=None,  # und 侧 merge（None=跟随 use_spatial_merge）
        share_unified2llm=True,   # gen/und 是否共用同一个 MLP
        use_mrope=False,           # 是否使用 3D MRoPE（Qwen3-VL backbone 时开启）
        # channel-wise loss weighting (RAE-style)
        sem_dim=None,              # semantic channels (前 N 维)，None 表示不拆分
        recon_loss_weight=1.0,     # recon 通道的 loss 权重
        normalize_channel_weight=False,  # 归一化使总 loss scale 不变
        # prediction target type
        pred_type='v_pred',        # 'v_pred': predict velocity; 'x_pred': predict clean with v supervision
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
        self.use_qwen3_unified = use_qwen3_unified
        self.use_spatial_merge = use_spatial_merge
        # 独立控制 gen/und merge，None 时跟随 use_spatial_merge
        self.use_spatial_merge_gen = use_spatial_merge if use_spatial_merge_gen is None else use_spatial_merge_gen
        self.use_spatial_merge_und = use_spatial_merge if use_spatial_merge_und is None else use_spatial_merge_und
        print(
            f"[UnimmConfig] use_qwen3_unified={self.use_qwen3_unified} "
            f"use_spatial_merge={self.use_spatial_merge} "
            f"use_spatial_merge_gen={self.use_spatial_merge_gen} "
            f"use_spatial_merge_und={self.use_spatial_merge_und} "
            f"max_latent_size={self.max_latent_size} "
            f"vit_max_num_patch_per_side={self.vit_max_num_patch_per_side}",
            flush=True
        )
        self.share_unified2llm = share_unified2llm
        self.use_mrope = use_mrope
        self.sem_dim = sem_dim
        self.recon_loss_weight = float(recon_loss_weight)
        self.normalize_channel_weight = normalize_channel_weight
        self.pred_type = pred_type


class UnimmMoT(PreTrainedModel):
    """MOT version of Unimm. Uses Qwen3VLMoTForConditionalGeneration as language model."""
    config_class = UnimmConfig
    base_model_prefix = 'unimm'

    def __init__(self, vl_model, vit_model, tokenizer, config: UnimmConfig):
        super().__init__(config)
        self.language_model = vl_model
        self.tokenizer = tokenizer
        # is_qwen3vl: llm_config 是嵌套结构（Qwen3VLConfig.text_config）时为 True
        # 与 use_qwen_vit 解耦，支持 Qwen3-VL backbone + use_qwen_vit=False 场景
        # Qwen3 text-only: text_config = llm_config（同一对象，flat）
        # Qwen3-VL: text_config 是独立的 Qwen3VLTextConfig（nested）
        self.is_qwen3vl = (
            hasattr(config.llm_config, 'text_config') and
            config.llm_config.text_config is not config.llm_config
        )
        # use_mrope: 由外部 --use_mrope 控制，与 use_qwen_vit 解耦
        self.use_mrope = config.use_mrope

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

        if config.use_qwen3_unified:
            # === Qwen3Unified 统一表征路径 ===
            # Qwen3Unified encoder 是外部模型（和 Flux VAE 一样），不作为子模块放在这里。
            # UnimmMoT 只持有投影层和 type embedding，encoder 在训练循环 forward() 前调用。
            self.vit_model = None
            self.connector = None
            unified_z = config.vae_config.z_channels   # 1280
            unified_downsample = config.vae_config.downsample  # 16
            self.latent_patch_size = 1
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = unified_downsample
            self.max_latent_size = config.max_latent_size
            self.latent_channel = unified_z
            self.patch_latent_dim = unified_z
            self.vit_patch_size = unified_downsample
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            # spatial merge factor per side
            merge_factor_gen = 4 if config.use_spatial_merge_gen else 1
            merge_factor_und = 4 if config.use_spatial_merge_und else 1
            # MLP: shared or separate for gen/und
            if config.share_unified2llm:
                # shared: use the larger merge factor to be compatible
                # (wenn beide gleich, kein Problem; bei Unterschied: separate benötigt)
                assert merge_factor_gen == merge_factor_und, \
                    "share_unified2llm=True requires same merge factor for gen and und"
                unified2llm_in = unified_z * merge_factor_gen
                self.unified2llm = MLPconnector(unified2llm_in, self.hidden_size, config.connector_act)
            else:
                # separate MLPs for gen and und
                self.unified2llm_und = MLPconnector(unified_z * merge_factor_und, self.hidden_size, config.connector_act)
                if config.visual_gen:
                    self.unified2llm_gen = MLPconnector(unified_z * merge_factor_gen, self.hidden_size, config.connector_act)
            # und 侧位置编码：use_mrope=True 时 3D RoPE 已编码空间位置，不需要 additive pos embed
            if config.visual_und and not self.use_mrope:
                self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
            else:
                self.vit_pos_embed = None
            # 生成侧专用模块：只在 visual_gen=True 时创建
            if config.visual_gen:
                self.llm2vae = nn.Linear(self.hidden_size, unified_z * merge_factor_gen)
                self.time_embedder = TimestepEmbedder(self.hidden_size)
                if not self.use_mrope:
                    self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)
        else:
            # === 原有路径：SigLIP + Flux VAE ===
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
                # MRoPE 启用时 pos embed 由 RoPE 层提供，无需额外 latent_pos_embed
                if not self.use_mrope:
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

    def rebuild_unified2llm_und_as_exact_merger(
        self,
        total_in: int,
        merger_hidden: int,
        out_dim: int,
        sem_per_patch: int,
    ) -> None:
        """Replace unified2llm_und with ExactMergerConnector of the given shape.

        Call this BEFORE load_state_dict when the checkpoint has ExactMergerConnector
        format (contains unified2llm_und.norm.weight). Ensures shape compatibility so
        load_state_dict correctly loads the weights.

        Args:
            total_in:       input dim = z_per_patch × 4, e.g. 5120
            merger_hidden:  intermediate dim = sem_per_patch × 4, e.g. 4608
            out_dim:        LLM hidden size, e.g. 4096
            sem_per_patch:  semantic channels per patch from official ViT, e.g. 1152
        """
        self.unified2llm_und = ExactMergerConnector(
            total_in=total_in,
            merger_hidden=merger_hidden,
            out_dim=out_dim,
            sem_per_patch=sem_per_patch,
        )

    @staticmethod
    def spatial_merge(x: torch.Tensor, h: int, w: int, merge_size: int = 2) -> torch.Tensor:
        """[N, C] → [N/merge_size^2, C*merge_size^2], row-major 2x2 block merge."""
        m = merge_size
        assert h % m == 0 and w % m == 0
        C = x.shape[-1]
        # reshape to [h, w, C] then block merge
        x = x.reshape(h, w, C)
        # [h/m, m, w/m, m, C] → [h/m, w/m, m*m*C]
        x = x.reshape(h // m, m, w // m, m, C)
        x = x.permute(0, 2, 1, 3, 4).reshape(h // m, w // m, m * m * C)
        return x.reshape((h // m) * (w // m), m * m * C)

    @staticmethod
    def spatial_unshuffle(x: torch.Tensor, h: int, w: int, merge_size: int = 2) -> torch.Tensor:
        """[N/merge_size^2, C*merge_size^2] → [N, C], inverse of spatial_merge."""
        m = merge_size
        mh, mw = h // m, w // m
        merged_C = x.shape[-1]
        C = merged_C // (m * m)
        # [mh*mw, m*m*C] → [mh, mw, m, m, C] → [h, w, C] → [N, C]
        x = x.reshape(mh, mw, m, m, C)
        x = x.permute(0, 2, 1, 3, 4).reshape(h, w, C)
        return x.reshape(h * w, C)

    @property
    def _embed_tokens(self):
        """Access embed_tokens."""
        return self.language_model.model.language_model.embed_tokens

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
        real_sequence_length: int = None,  # padding 前的真实序列长度（仅 flex 模式下有值）
        # pre-computed deepstack for use_qwen3_unified+visual_und path (qwen3vl only)
        deepstack_visual_embeds: Optional[list] = None,
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
        packed_text_embedding = self._embed_tokens(packed_text_ids)
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
            # Use padded sequence_length (not sum(sample_lens)) so BlockMask shape
            # is constant across steps — avoids torch.compile recompilation when
            # the real sequence length varies.
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads,
                Q_LEN=sequence_length, KV_LEN=sequence_length,
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask

        # === 视觉理解分支 ===
        if self.config.visual_und:
            if self.config.use_qwen3_unified:
                # --- Qwen3Unified 理解路径 ---
                # packed_vit_tokens: [total_und_tokens, 1280]
                # 由训练循环在 fsdp_model(**data) 前 encode + normalize 好，直接传入
                _und_mlp = self.unified2llm if self.config.share_unified2llm else self.unified2llm_und
                # 无论有无图片，统一走相同代码路径（不用 if/else 分支），
                # 保证 unified2llm_und 的 FSDP reduce-scatter 在所有 rank 上
                # 于 language_model backward 中途同一位置触发，避免 HCCL 死锁。
                _has_real_vit = packed_vit_tokens is not None and packed_vit_token_indexes is not None
                if not _has_real_vit:
                    # 纯文本 rank：创建 dummy VIT token，走相同代码路径
                    packed_vit_tokens = torch.zeros(
                        1, _und_mlp.fc1.in_features,
                        device=packed_sequence.device, dtype=packed_sequence.dtype)
                    packed_vit_token_indexes = torch.zeros(
                        1, dtype=torch.long, device=packed_sequence.device)
                packed_vit_token_embed = _und_mlp(packed_vit_tokens).to(packed_sequence.dtype)
                if self.vit_pos_embed is not None:
                    if packed_vit_position_ids is not None:
                        vit_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                    else:
                        # 纯文本 batch：dummy forward，保证所有 rank 计算图结构一致，
                        # 避免 backward 时 HCCL 通信顺序不同导致死锁。
                        # pos_embed 是 frozen 的，乘 0 不影响梯度更新，只保留图结构。
                        _dummy_pos_ids = torch.zeros(1, dtype=torch.long, device=packed_sequence.device)
                        vit_pos_emb = self.vit_pos_embed(_dummy_pos_ids) * 0
                    packed_vit_token_embed = packed_vit_token_embed + vit_pos_emb
                if _has_real_vit:
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
                else:
                    # dummy：in-place 加 0，不改变值，但 backward 图结构与有图 rank 完全一致
                    packed_sequence[packed_vit_token_indexes] = (
                        packed_sequence[packed_vit_token_indexes] + packed_vit_token_embed * 0
                    )
                vit_extra_inputs = {}
                # deepstack: build visual_pos_masks and pass pre-computed embeds (qwen3vl only)
                if deepstack_visual_embeds is not None and packed_vit_token_indexes is not None:
                    _vis_mask = torch.zeros(sequence_length, dtype=torch.bool,
                                            device=packed_sequence.device)
                    _vis_mask[packed_vit_token_indexes] = True
                    vit_extra_inputs['visual_pos_masks'] = _vis_mask
                    vit_extra_inputs['deepstack_visual_embeds'] = deepstack_visual_embeds

            elif self.config.use_qwen_vit:
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
                    packed_vit_token_embed = self._embed_tokens(image_token_ids)

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
        z_latent_for_loss = None   # noisy latent in prediction space, set below for x_pred
        t_for_vae_tokens = None    # timesteps in prediction space, set below for x_pred
        if self.config.visual_gen:
            if self.config.use_qwen3_unified:
                # --- Qwen3Unified 生成路径 ---
                # padded_latent: [B, z_channels, h_max, w_max]，train loop encode 好后传入
                if padded_latent is not None and packed_vae_token_indexes is not None:
                    # unpad + pack: [B, C, h, w] -> [N_total, C]
                    packed_latent_list = []
                    for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                        packed_latent_list.append(
                            latent[:, :h, :w].permute(1, 2, 0).reshape(-1, self.latent_channel)
                        )
                    packed_latent_clean = torch.cat(packed_latent_list, dim=0)
                    noise = torch.randn_like(packed_latent_clean)
                    # noising in original token space [N, C]
                    packed_latent = (1 - packed_timesteps[:, None]) * noise + packed_timesteps[:, None] * packed_latent_clean
                    if self.config.use_spatial_merge_gen:
                        # merge after noising: each image separately [N, C] -> [N/4, C*4]
                        merged_list = []
                        offset = 0
                        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                            n = h * w
                            merged_list.append(self.spatial_merge(packed_latent[offset:offset+n], h, w))
                            offset += n
                        packed_latent = torch.cat(merged_list, dim=0)
                        z_latent_for_loss = packed_latent          # [N/4, 4C] merged noisy, for x_pred loss
                        # timesteps: N -> N/4, each merge block shares the same timestep
                        packed_timesteps_merged = packed_timesteps.reshape(-1, 4)[:, 0]
                        t_for_vae_tokens = packed_timesteps_merged # [N/4]
                        packed_timestep_embeds = self.time_embedder(packed_timesteps_merged)
                    else:
                        z_latent_for_loss = packed_latent          # [N, C] original noisy, for x_pred loss
                        t_for_vae_tokens = packed_timesteps        # [N]
                        packed_timestep_embeds = self.time_embedder(packed_timesteps)
                    _gen_mlp = self.unified2llm if self.config.share_unified2llm else self.unified2llm_gen
                    packed_latent = _gen_mlp(packed_latent) + packed_timestep_embeds
                    if not self.use_mrope:
                        latent_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                        packed_latent = packed_latent + latent_pos_emb
                    packed_sequence[packed_vae_token_indexes] = packed_latent
                else:
                    device = packed_sequence.device
                    dummy_t = torch.zeros(1, device=device)
                    dummy_out = self.time_embedder(dummy_t.unsqueeze(0))
                    packed_sequence = packed_sequence + dummy_out.sum() * 0
            else:
                # --- 原有 Flux VAE 生成路径 ---
                if padded_latent is not None and packed_vae_token_indexes is not None:
                    p = self.latent_patch_size
                    packed_latent = []
                    for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                        latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                        packed_latent.append(latent)
                    packed_latent_clean = torch.cat(packed_latent, dim=0)
                    noise = torch.randn_like(packed_latent_clean)
                    packed_latent = (1 - packed_timesteps[:, None]) * noise + packed_timesteps[:, None] * packed_latent_clean
                    z_latent_for_loss = packed_latent              # [N, D] patchified noisy, for x_pred loss
                    t_for_vae_tokens = packed_timesteps            # [N]
                    packed_timestep_embeds = self.time_embedder(packed_timesteps)
                    packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds
                    if not self.use_mrope:
                        latent_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                        packed_latent = packed_latent + latent_pos_emb
                    packed_sequence[packed_vae_token_indexes] = packed_latent
                else:
                    # TimestepEmbedder FSDP dummy forward
                    device = packed_sequence.device
                    dummy_t = torch.zeros(1, device=device)
                    dummy_out = self.time_embedder(dummy_t.unsqueeze(0))
                    packed_sequence = packed_sequence + dummy_out.sum() * 0

        # === 准备 language_model 的输入 ===
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            # vit tokens 属于 und 侧：SigLIP 路径和 Qwen3Unified 路径都需要合并
            if (not self.config.use_qwen_vit or self.config.use_qwen3_unified) and packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)

            # mark_dynamic on index tensors: tells dynamo to use symbolic shapes
            # for dim=0 of und/gen indexes. This way, ONE compiled graph handles
            # all index lengths — no recompilation when gen/und token counts change.
            #
            # We do NOT use torch.compile(dynamic=True) globally because it makes
            # flex_attention's Q/K/V sequence dim symbolic, causing PT 2.5 inductor
            # to fail (NoValidChoicesError for flex_attention Triton kernel).
            #
            # mark_dynamic is selective: only the index tensors get symbolic dim=0.
            # packed_sequence (5120) stays static → flex_attention works fine.
            sink_index = sequence_length - 1
            if packed_und_token_indexes is None or packed_und_token_indexes.numel() == 0:
                packed_und_token_indexes = packed_sequence.new_full(
                    (_SINK_LEN,), sink_index, dtype=torch.long)
            if packed_vae_token_indexes is None or packed_vae_token_indexes.numel() == 0:
                packed_vae_token_indexes = packed_sequence.new_full(
                    (_SINK_LEN,), sink_index, dtype=torch.long)
            torch._dynamo.mark_dynamic(packed_und_token_indexes, 0)
            torch._dynamo.mark_dynamic(packed_vae_token_indexes, 0)

            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        # 如果使用 Qwen VIT,传递视觉相关参数
        if self.config.use_qwen_vit and self.config.visual_und:
            extra_inputs.update(vit_extra_inputs)
            # 传递 input_ids 给 Qwen3VL 用于 get_placeholder_mask
            if input_ids is not None:
                extra_inputs['input_ids'] = input_ids
        # use_qwen3_unified: pass deepstack if available (qwen3vl only)
        elif self.config.use_qwen3_unified and self.config.visual_und and vit_extra_inputs:
            extra_inputs.update(vit_extra_inputs)

        # === 调用 language_model ===
        # 注意: 训练时走 forward_train, 参数名是 packed_sequence / packed_position_ids
        # 推理时走 forward_inference, 参数名是 packed_query_sequence / packed_query_position_ids
        # self.language_model.forward() 会根据 training 状态自动分发
        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        # === 损失计算 ===
        mse = None
        mse_sem = None
        mse_recon = None
        if self.config.visual_gen and padded_latent is not None and mse_loss_indexes is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            # target = clean - noise (velocity)
            target = packed_latent_clean - noise  # [N, C] in original token space
            if self.config.use_spatial_merge_gen:
                # target is in original N space; merge to match preds in N/4 space
                merged_target_list = []
                offset = 0
                for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                    n = h * w
                    merged_target_list.append(self.spatial_merge(target[offset:offset+n], h, w))
                    offset += n
                target_merged = torch.cat(merged_target_list, dim=0)
                # mse_loss_indexes is in merged space
                vae_abs_to_rel = {v.item(): i for i, v in enumerate(packed_vae_token_indexes)}
                rel_indexes = torch.tensor(
                    [vae_abs_to_rel[m.item()] for m in mse_loss_indexes],
                    device=target.device, dtype=torch.long
                )
                mse_target = target_merged[rel_indexes]
            else:
                vae_abs_to_rel = {v.item(): i for i, v in enumerate(packed_vae_token_indexes)}
                rel_indexes = torch.tensor(
                    [vae_abs_to_rel[m.item()] for m in mse_loss_indexes],
                    device=target.device, dtype=torch.long
                )
                mse_target = target[rel_indexes]

            if mse_target.numel() > 0:
                _pred_type = getattr(self.config, 'pred_type', 'v_pred')
                _T_EPS = 5e-2  # clamp (1-t) to avoid division by zero near t=1; align with JiT
                if self.config.use_spatial_merge_gen:
                    # unshuffle preds and target back to original [N, C] for sem/recon split
                    preds_list, target_list = [], []
                    p_offset, t_offset = 0, 0
                    if _pred_type == 'x_pred':
                        # z_for_loss: merged noisy latent for loss tokens [M_merged, 4C]
                        z_for_loss = z_latent_for_loss[rel_indexes]
                        z_offset = 0
                    for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                        n_merged = (h // 2) * (w // 2)
                        preds_orig = self.spatial_unshuffle(packed_mse_preds[p_offset:p_offset+n_merged], h, w)
                        target_orig = self.spatial_unshuffle(mse_target[t_offset:t_offset+n_merged], h, w)
                        if _pred_type == 'x_pred':
                            # convert x_pred → v_pred, then supervise with v_true
                            z_orig = self.spatial_unshuffle(z_for_loss[z_offset:z_offset+n_merged], h, w)
                            t_img = float(t_for_vae_tokens[rel_indexes[z_offset]])
                            v_pred_img = (preds_orig - z_orig) / max(1.0 - t_img, _T_EPS)
                            preds_list.append(v_pred_img)
                            z_offset += n_merged
                        else:
                            preds_list.append(preds_orig)
                        target_list.append(target_orig)
                        p_offset += n_merged
                        t_offset += n_merged
                    mse_raw = (torch.cat(preds_list, dim=0) - torch.cat(target_list, dim=0)) ** 2  # [N, C]
                else:
                    if _pred_type == 'x_pred':
                        # z_for_loss: original-space noisy latent for loss tokens [M, C]
                        z_for_loss = z_latent_for_loss[rel_indexes]
                        t_for_loss = t_for_vae_tokens[rel_indexes].float()  # [M]
                        v_pred = (packed_mse_preds - z_for_loss) / (1 - t_for_loss[:, None]).clamp_min(_T_EPS)
                        mse_raw = (v_pred - mse_target) ** 2  # [M, C]
                    else:
                        mse_raw = (packed_mse_preds - mse_target) ** 2  # [N, C]
                sem_dim = self.config.sem_dim
                mse_sem = mse_recon = None
                if sem_dim is not None and sem_dim > 0:
                    mse_sem   = mse_raw[..., :sem_dim]    # [N, sem_dim]
                    mse_recon = mse_raw[..., sem_dim:]     # [N, recon_dim]
                    recon_w = self.config.recon_loss_weight
                    if recon_w != 1.0:
                        C = mse_raw.shape[-1]
                        recon_dim = C - sem_dim
                        sem_w = 1.0
                        if self.config.normalize_channel_weight:
                            norm_factor = C / (sem_dim * sem_w + recon_dim * recon_w)
                            sem_w   *= norm_factor
                            recon_w *= norm_factor
                        weight = torch.ones(1, C, device=mse_raw.device)
                        weight[..., :sem_dim]  = sem_w
                        weight[..., sem_dim:]  = recon_w
                        mse_raw = mse_raw * weight
                mse = mse_raw
            else:
                mse = mse_sem = mse_recon = torch.tensor(0.0, device=packed_mse_preds.device)

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce, mse_sem=mse_sem, mse_recon=mse_recon)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, raw=False):
        """Encode text prompts into KV-cache-ready format.

        Args:
            raw: If True, encode each prompt as ``[bos]{text}[eos]`` without
                 auto-wrapping with ``user\\n`` / ``assistant\\n`` prefixes.
                 Useful for editing tasks where user/assistant segments are
                 managed by the caller.
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
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                'start_of_image': tokenizer.encode("<|vision_start|>")[0],
                'end_of_image': tokenizer.encode("<|vision_end|>")[0],
            }

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            if raw:
                # Raw mode: [bos]{text}[eos], caller manages user/assistant prefixes
                prompt_ids = tokenizer.encode(prompt)
                text_ids = (
                    [new_token_ids['bos_token_id']] + prompt_ids + [new_token_ids['eos_token_id']]
                )
            else:
                # Match T2I training format:
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
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self._embed_tokens(packed_text_ids)

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
            # new_rope: ViT 区域 rope_id 推进 (M+2)，与训练 dataset_base.py:872-873 一致
            new_rope.append(curr_position_id + num_img_tokens + 2)

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
        packed_text_embedding = self._embed_tokens(packed_text_ids)
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
                    packed_vit_token_embed = self._embed_tokens(image_token_ids)
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
            sample_lens=packed_seqlens,
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

    @torch.no_grad
    def forward_cache_update_unified_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        # pre-computed deepstack features from unified VIT (qwen3vl only, optional)
        deepstack_visual_embeds: Optional[list] = None,
        is_causal: bool = False,  # False: image-only full attn (Qwen3); True: full user turn causal (Qwen3-VL)
    ):
        """Qwen3Unified 理解路径的 VIT prefill: RAE tokens → unified2llm_und → KV cache."""
        packed_text_embedding = self._embed_tokens(packed_text_ids)
        seq_len = sum(packed_seqlens)
        packed_sequence = packed_text_embedding.new_zeros((seq_len, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        _und_mlp = self.unified2llm if self.config.share_unified2llm else self.unified2llm_und
        packed_vit_token_embed = _und_mlp(packed_vit_tokens).to(packed_sequence.dtype)
        if self.vit_pos_embed is not None and packed_vit_position_ids is not None:
            vit_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_pos_emb
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        # deepstack: build visual_pos_masks and pass (qwen3vl only, no-op when None)
        if deepstack_visual_embeds is not None:
            visual_pos_masks = torch.zeros(seq_len, dtype=torch.bool,
                                           device=packed_sequence.device)
            visual_pos_masks[packed_vit_token_indexes] = True
            extra_inputs['deepstack_visual_embeds'] = deepstack_visual_embeds
            extra_inputs['visual_pos_masks'] = visual_pos_masks

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            sample_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=is_causal,
            **extra_inputs,
        )
        return output.past_key_values

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
                # (与训练一致：训练时 3D position_ids dim 0 = 常数 st_idx)
                st_idx = curr_position_id
                pos_1d = torch.full((num_img_tokens + 2,), st_idx, dtype=torch.long)
                packed_position_ids.append(pos_1d)

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            # new_rope: 条件图 (loss=0) rope_id 只 +1，与训练 dataset_base.py:715 一致
            # （目标图 loss=1 用 max(st_idx+h, st_idx+w)+1，但 prepare_vae_images 只用于条件图）
            new_rope.append(st_idx + 1)

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
        packed_text_embedding = self._embed_tokens(packed_text_ids)
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
            sample_lens=packed_seqlens,
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
        patchified_vae_latent_shapes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            _pos_downsample = (
                self.latent_downsample * 2
                if (self.config.use_qwen3_unified and self.config.use_spatial_merge_gen)
                else self.latent_downsample
            )
            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                _pos_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))
            if self.config.use_qwen3_unified and self.config.use_spatial_merge_gen:
                num_image_tokens = (h // 2) * (w // 2)
                # 初始噪声直接在 merge 后空间生成：[N/4, C*4]
                packed_init_noises.append(
                    torch.randn(
                        num_image_tokens, self.latent_channel * 4,
                        generator=generator,
                    )
                )
            else:
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
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
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
            if self.config.use_qwen3_unified and self.config.use_spatial_merge_gen:
                num_image_tokens = (h // 2) * (w // 2)
            else:
                num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            st_idx = curr_position_id
            if self.is_qwen3vl:
                # 3D MRoPE: 与 prepare_vae_latent 保持一致
                t_pos = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                h_grid, w_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                h_pos = torch.cat([torch.tensor([st_idx]), h_grid.flatten() + st_idx, torch.tensor([st_idx + h - 1])])
                w_pos = torch.cat([torch.tensor([st_idx]), w_grid.flatten() + st_idx, torch.tensor([st_idx + w - 1])])
                packed_position_ids.append(torch.stack([t_pos, h_pos, w_pos], dim=0))  # [3, N+2]
            else:
                pos_1d = torch.full((num_image_tokens + 2,), st_idx, dtype=torch.long)
                packed_position_ids.append(pos_1d)

        if self.is_qwen3vl:
            full_position_ids_3d = torch.zeros(3, query_curr, dtype=torch.long)
            pos_offset = 0
            for pos_3d in packed_position_ids:
                seg_len = pos_3d.shape[1]
                full_position_ids_3d[:, pos_offset:pos_offset + seg_len] = pos_3d
                pos_offset += seg_len
            cfg_pos = full_position_ids_3d
        else:
            cfg_pos = torch.cat(packed_position_ids, dim=0)

        generation_input = {
            "cfg_packed_position_ids": cfg_pos,
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
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        max_t: float = 1.0,
        return_intermediates: bool = False,
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
        prompt_token_lens = 0  # 默认值，flex_attention 路径中也需要使用

        if inference_mode == "flex":
            # 根据 attention implementation 创建对应格式的掩码
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
                # flex_attention: 使用 block_mask（稀疏压缩格式）
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

        timesteps = torch.linspace(0, max_t, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[1:]
        _x_pred_mode = getattr(self.config, 'pred_type', 'v_pred') == 'x_pred'
        intermediate_latents = []  # [(step_idx, t_val, x_t_snapshot), ...]
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # x_pred: clamp query timestep so (1-t_model) >= |dts[i]|
            # prevents (x_pred - x_t)/(1-t) singularity when t=1.0 at the last step.
            # At the last step this makes x_t move exactly to x_pred (same as JiT).
            if _x_pred_mode:
                t_model = float(t.clamp(max=1.0 - abs(float(dts[i]))))
            else:
                t_model = float(t)
            timestep = torch.tensor([t_model] * x_t.shape[0], device=x_t.device)
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
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
            if return_intermediates:
                intermediate_latents.append((i, float(t), x_t.detach().clone()))

        # if enable_taylorseer:
        #     del model_pred_cache_dic, model_pred_current
        #     del model_pred_text_cache_dic, model_pred_text_current
        #     del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - prompt_token_lens - 2).tolist())
        if return_intermediates:
            return unpacked_latent, intermediate_latents
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
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
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
        packed_text_embedding = self._embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        x_t_raw = x_t  # save before projection for x_pred conversion
        # Qwen3 text-only: 1D RoPE 无法区分 VAE 空间位置，需要 latent_pos_embed
        # Qwen3-VL: 3D MRoPE 已在 RoPE 层提供空间位置信息，无需额外 pos embed
        packed_timestep_embeds = self.time_embedder(timestep)
        if self.config.use_qwen3_unified:
            _gen_mlp = self.unified2llm if self.config.share_unified2llm else self.unified2llm_gen
            x_t = _gen_mlp(x_t) + packed_timestep_embeds
        else:
            x_t = self.vae2llm(x_t) + packed_timestep_embeds
        if not self.use_mrope:
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
        _pred_type = getattr(self.config, 'pred_type', 'v_pred')
        _t_val = float(timestep[0])
        _T_EPS = 5e-2  # match training clamp to avoid OOD amplification near t=1

        def _to_velocity(model_out_vae):
            """Convert model output to velocity. x_pred: (x_pred - x_t_raw)/(1-t). v_pred: identity."""
            if _pred_type == 'x_pred':
                x_raw = x_t_raw.to(model_out_vae.dtype)
                return (model_out_vae - x_raw) / max(1.0 - _t_val, _T_EPS)
            return model_out_vae

        v_t = _to_velocity(self.llm2vae(output.last_hidden_state)[packed_vae_token_indexes])
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
            cfg_packed_vae_token_indexes = packed_vae_token_indexes - (packed_vae_token_indexes[0] - 1)
            cfg_text_v_t = _to_velocity(self.llm2vae(cfg_text_output.last_hidden_state)[cfg_packed_vae_token_indexes])

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                sample_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = _to_velocity(self.llm2vae(cfg_img_output.last_hidden_state)[packed_vae_token_indexes])

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
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self._embed_tokens(curr_tokens)
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
                sample_lens=query_lens,
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
                logits = pred_logits.clone()

                # 1. repetition_penalty (在 temperature 之前，与 HF 一致)
                if repetition_penalty != 1.0 and generated_sequence:
                    prev_tokens = torch.cat(generated_sequence, dim=0).unique()
                    score = logits[0, prev_tokens]
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[0, prev_tokens] = score

                # 2. presence_penalty
                if presence_penalty != 0.0 and generated_sequence:
                    prev_tokens = torch.cat(generated_sequence, dim=0).unique()
                    logits[0, prev_tokens] -= presence_penalty

                # 3. temperature
                logits = logits / temperature

                # 4. top_k
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < top_k_vals[..., -1:]] = float('-inf')

                # 5. top_p (与 HF TopPLogitsWarper 对齐)
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    sorted_indices_to_remove[..., -1:] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                probs = nn.functional.softmax(logits, dim=-1)
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

