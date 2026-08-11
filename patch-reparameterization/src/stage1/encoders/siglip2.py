from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import SiglipModel, SiglipConfig, SiglipVisionModel

import logging
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _log_recon_diff_visualization(diff_tensor, step=None, token_indices=None, max_dims_display=768):
    """逐样本逐维度逐 token 的 diff 打印与可视化，并保存到 wandb（复用 wandb_utils）。

    Args:
        diff_tensor: (B, N, D) recon_post - origin_recon_post
        step: wandb step
        token_indices: 要详细打印的 token 索引列表，默认 [0, 1, 50, 100, 120, -1]
        max_dims_display: 热力图显示的维度数（若 D 很大则下采样）
    """
    if not logger.isEnabledFor(logging.DEBUG) and not _HAS_MATPLOTLIB:
        return
    B, N, D = diff_tensor.shape
    diff_np = diff_tensor.detach().float().cpu().numpy()
    abs_diff_np = np.abs(diff_np)
    token_indices = token_indices or [0, 1, 50, 100, 120, 200, -1]
    token_indices = [t if t >= 0 else N + t for t in token_indices]
    token_indices = [t for t in token_indices if t < N]

    # 1. 详细打印：每个样本、每个 token、top-k 维度
    lines = ["\n========== recon_diff 详细数据 (sample, token, top_dims) =========="]
    for b in range(min(B, 4)):
        for tok_idx in token_indices:
            row = abs_diff_np[b, tok_idx]
            top_k = min(5, len(row))
            top_indices = np.argsort(row)[-top_k:][::-1]
            top_vals = row[top_indices]
            dim_str = ", ".join([f"d{d}= {v:.4f}" for d, v in zip(top_indices, top_vals)])
            lines.append(f"  sample{b} token{tok_idx}: max_dim={top_indices[0]} ({top_vals[0]:.4f}) | top5: {dim_str}")
    logger.debug("\n".join(lines))

    if not _HAS_MATPLOTLIB:
        return

    figs = {}
    # 2. 热力图：sample x token，值为该 (sample,token) 的 max_abs_diff
    max_per_st = abs_diff_np.max(axis=-1)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    im1 = ax1.imshow(max_per_st, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Token")
    ax1.set_ylabel("Sample")
    ax1.set_title("recon_diff: max |diff| per (sample, token)")
    plt.colorbar(im1, ax=ax1)
    plt.tight_layout()
    figs["diff_heatmap_sample_x_token"] = fig1

    # 2b. 热力图：sample x token，值为该 (sample,token) 的 mean_abs_diff（每个样本每个 token 的维度平均 diff）
    mean_per_st = abs_diff_np.mean(axis=-1)  # (B, N)
    fig1b, ax1b = plt.subplots(figsize=(10, 4))
    im1b = ax1b.imshow(mean_per_st, aspect="auto", cmap="viridis")
    ax1b.set_xlabel("Token")
    ax1b.set_ylabel("Sample")
    ax1b.set_title("recon_diff: mean |diff| per (sample, token) over dims")
    plt.colorbar(im1b, ax=ax1b)
    plt.tight_layout()
    figs["diff_heatmap_sample_x_token_mean"] = fig1b

    # 3. 热力图：token x dim，batch 平均
    mean_abs = abs_diff_np.mean(axis=0)
    dim_step = max(1, D // max_dims_display)
    mean_subsample = mean_abs[:, ::dim_step]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    im2 = ax2.imshow(mean_subsample.T, aspect="auto", cmap="viridis", origin="lower")
    ax2.set_xlabel("Token")
    ax2.set_ylabel(f"Dim (step={dim_step})")
    ax2.set_title("recon_diff: mean |diff| over batch (token x dim)")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    figs["diff_heatmap_token_x_dim"] = fig2

    # 4. 热力图：sample x dim，针对 token 0, 1, 120
    for tok_idx in [0, 1, 120]:
        if tok_idx >= N:
            continue
        mat = abs_diff_np[:, tok_idx, ::dim_step]
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xlabel(f"Dim (step={dim_step})")
        ax.set_ylabel("Sample")
        ax.set_title(f"recon_diff: |diff| for token {tok_idx} (sample x dim)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        figs[f"diff_heatmap_tok{tok_idx}_sample_x_dim"] = fig

    try:
        from utils.wandb_utils import log_matplotlib_figures
        log_matplotlib_figures(figs, prefix="encoder/", step=step)
    except ImportError:
        logger.warning("wandb_utils not found, skipping matplotlib figures logging")
        for fig in figs.values():
            plt.close(fig)
    except Exception as e:
        logger.warning(f"log_matplotlib_figures failed: {e}")
        for fig in figs.values():
            plt.close(fig)


def _log_recon_cosine_similarity(new_recon_post, origin_recon_post, step=None, token_indices=None):
    """逐样本、逐 token 的 new_recon_post 与 origin_recon_post 的 cosine 相似度：log + wandb 可视化。

    Args:
        new_recon_post: (B, N, D) merged 后 recon post
        origin_recon_post: (B, N, D) 原始 recon post
        step: wandb step
        token_indices: 要详细打印的 token 索引列表
    """
    if not logger.isEnabledFor(logging.DEBUG) and not _HAS_MATPLOTLIB:
        return
    B, N, D = new_recon_post.shape
    a = new_recon_post.detach().float()
    b = origin_recon_post.detach().float()
    # (B, N) per (sample, token) cosine similarity
    norm_a = a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    norm_b = b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (a * b).sum(dim=-1) / (norm_a.squeeze(-1) * norm_b.squeeze(-1) + 1e-8)
    cos_np = cos_sim.cpu().numpy()

    token_indices = token_indices or [0, 1, 50, 100, 120, 200, -1]
    token_indices = [t if t >= 0 else N + t for t in token_indices]
    token_indices = [t for t in token_indices if t < N]

    # 1. 逐样本：该样本所有 token 的 cos 均值/最小/最大
    lines = ["\n========== recon cosine similarity (new vs origin) per sample & per token =========="]
    for b in range(B):
        row = cos_np[b]
        lines.append(f"  sample{b}: mean={row.mean():.6f} min={row.min():.6f} max={row.max():.6f}")
    # 2. 逐 token：所有样本该 token 的 cos 均值/最小/最大
    lines.append("  --- per token (over samples) ---")
    for tok_idx in token_indices:
        col = cos_np[:, tok_idx]
        lines.append(f"  token{tok_idx}: mean={col.mean():.6f} min={col.min():.6f} max={col.max():.6f}")
    # 3. 每个样本 cos 最小的 token
    for b in range(min(B, 4)):
        min_tok = cos_np[b].argmin()
        lines.append(f"  sample{b} min_cos token={min_tok} value={cos_np[b, min_tok]:.6f}")
    logger.debug("\n".join(lines))

    if not _HAS_MATPLOTLIB:
        return

    figs = {}
    # 热力图：sample x token，值为 cosine similarity
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    im1 = ax1.imshow(cos_np, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax1.set_xlabel("Token")
    ax1.set_ylabel("Sample")
    ax1.set_title("recon_cos_sim: new vs origin per (sample, token)")
    plt.colorbar(im1, ax=ax1)
    plt.tight_layout()
    figs["cos_sim_heatmap_sample_x_token"] = fig1

    # 折线：每个样本的 cos 随 token 变化（仅前几个 sample）
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for b in range(min(B, 4)):
        ax2.plot(cos_np[b], label=f"sample{b}", alpha=0.8)
    ax2.set_xlabel("Token")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("recon_cos_sim: cos vs token (per sample)")
    ax2.legend()
    ax2.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    figs["cos_sim_line_per_sample"] = fig2

    # 折线：每个 token 的 cos 随 sample 变化（选若干 token）
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    for tok_idx in token_indices[:6]:
        ax3.plot(cos_np[:, tok_idx], label=f"token{tok_idx}", alpha=0.8)
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Cosine similarity")
    ax3.set_title("recon_cos_sim: cos vs sample (per token)")
    ax3.legend()
    ax3.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    figs["cos_sim_line_per_token"] = fig3

    try:
        from utils.wandb_utils import log_matplotlib_figures
        log_matplotlib_figures(figs, prefix="encoder/", step=step)
    except ImportError:
        logger.warning("wandb_utils not found, skipping cos_sim figures logging")
        for fig in figs.values():
            plt.close(fig)
    except Exception as e:
        logger.warning(f"log_matplotlib_figures (cos_sim) failed: {e}")
        for fig in figs.values():
            plt.close(fig)


class ReconAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 关键的 Scale 和 Shift 参数
        # 初始化技巧：Scale 初始化为 1，Shift 初始化为 0，让训练平滑开始
        self.pre_scale = nn.Parameter(torch.ones(dim))
        self.pre_shift = nn.Parameter(torch.zeros(dim))

        # 2. Decoder 专用的 LayerNorm
        self.recon_ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # x: [Batch, Seq, Dim] - 这是 Encoder 输出的 isotropic 特征

        # 步骤 A: 显式制造各向异性 (Scale & Shift)
        # 这里实际上是在做特征重加权 (Feature Re-weighting)
        x_modulated = x * self.pre_scale + self.pre_shift
        # logger.debug(f"x_modulated shape: {x_modulated.shape}")
        # logger.debug(f"pre_scale: {self.pre_scale}")
        # logger.debug(f"pre_shift: {self.pre_shift}")
        # 步骤 B: 通过 LayerNorm 固化这种分布
        # LN 会把 Scale 较小的维度压抑掉，只保留 Scale 大的“主成分”
        x_recon = self.recon_ln(x_modulated)

        return x_recon

class LowRankAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=68): # 瓶颈设小一点，比如 32 或 64
        super().__init__()
        # 先压缩：强迫模型把分散的信息挤压到少数几个维度
        self.down = nn.Linear(dim, bottleneck_dim)
        # 再放大：解压回原维度
        self.up = nn.Linear(bottleneck_dim, dim)
        self.act = nn.GELU() # 加个非线性，更强

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        delta = self.up(self.act(self.down(x)))
        return self.ln(x + delta)

class LowRankAdapterStopGrad(nn.Module):
    def __init__(self, dim, bottleneck_dim=68): # 瓶颈设小一点，比如 32 或 64
        super().__init__()
        # 先压缩：强迫模型把分散的信息挤压到少数几个维度
        self.down = nn.Linear(dim, bottleneck_dim)
        # 再放大：解压回原维度
        self.up = nn.Linear(bottleneck_dim, dim)
        self.act = nn.GELU() # 加个非线性，更强

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        delta = self.up(self.act(self.down(x)))
        return self.ln(x.detach() + delta)

class LowRankAdapterPreNorm(nn.Module):
    def __init__(self, dim, bottleneck_dim=68): # 瓶颈设小一点，比如 32 或 64
        super().__init__()
        # 先压缩：强迫模型把分散的信息挤压到少数几个维度
        self.down = nn.Linear(dim, bottleneck_dim)
        # 再放大：解压回原维度
        self.up = nn.Linear(bottleneck_dim, dim)
        self.act = nn.GELU() # 加个非线性，更强

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        delta = self.up(self.act(self.down(self.ln(x))))
        return x + delta

class LowRankAdapterNoNorm(nn.Module):
    def __init__(self, dim, bottleneck_dim=68): # 瓶颈设小一点，比如 32 或 64
        super().__init__()
        # 先压缩：强迫模型把分散的信息挤压到少数几个维度
        self.down = nn.Linear(dim, bottleneck_dim)
        # 再放大：解压回原维度
        self.up = nn.Linear(bottleneck_dim, dim)
        self.act = nn.GELU() # 加个非线性，更强


    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        delta = self.up(self.act(self.down(x)))
        return x + delta

class FullRankAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim, dim)
        self.up = nn.Linear(dim, dim)
        self.act = nn.GELU() # 加个非线性，更强

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        delta = self.up(self.act(self.down(x)))
        return self.ln(x + delta)

class FullRankAdapterPreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim, dim)
        self.up = nn.Linear(dim, dim)
        self.act = nn.GELU() # 加个非线性，更强
        self.ln_pre = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 残差连接：学的是“重建所需的增量”
        # 这里的 down->act->up 会天然制造出极强的各向异性
        x = self.ln_pre(x)
        delta = self.up(self.act(self.down(x)))
        return self.ln(x + delta)

class LinearAdapter(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        in_dim = kwargs.get("in_dim", dim)
        out_dim = kwargs.get("out_dim", dim)
        self.linear = nn.Linear(in_dim, out_dim)
        if kwargs.get("pre_norm", False):
            self.pre_ln = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.pre_ln = None

        if kwargs.get("post_norm", False):
            self.post_ln = nn.LayerNorm(dim)
        if kwargs.get("parameter_free_post_norm", False):
            self.post_ln = nn.LayerNorm(dim, elementwise_affine=False)
            logger.info(f"LinearAdapter using parameter_free_post_norm")
        if kwargs.get("no_post_norm", False):
            self.post_ln = None
    def forward(self, x):
        if self.pre_ln is not None:
            x = self.pre_ln(x)
        delta = self.linear(x)
        if hasattr(self, "post_ln") and self.post_ln is not None:
            return self.post_ln(delta)
        else:
            return delta

class MLPAdapter(nn.Module):
    """MLP adapter，层数由 kwargs.mlp_depth 控制，默认为 2 层 (in->hidden->out)"""
    def __init__(self, dim=None, **kwargs):
        super().__init__()
        self.dim = dim
        in_dim = kwargs.get("in_dim", dim)
        out_dim = kwargs.get("out_dim", dim)
        hidden_dim = kwargs.get("hidden_dim", dim)
        mlp_depth = kwargs.get("mlp_depth", 2)  # 默认 2 层，保持与原实现一致

        modules = []
        for i in range(mlp_depth):
            if i == 0:
                modules.append(nn.Linear(in_dim, hidden_dim))
            elif i == mlp_depth - 1:
                modules.append(nn.Linear(hidden_dim, out_dim))
            else:
                modules.append(nn.Linear(hidden_dim, hidden_dim))
            if i < mlp_depth - 1:
                modules.append(nn.GELU())
        self.mlp = nn.Sequential(*modules)
        self._last_linear_idx = 2 * (mlp_depth - 1)  # 最后一个 Linear 在 Sequential 中的索引

        if kwargs.get("pre_norm", False):
            self.pre_ln = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.pre_ln = None

        self.post_ln = None
        if kwargs.get("post_norm", False):
            self.post_ln = nn.LayerNorm(out_dim)
        if kwargs.get("parameter_free_post_norm", False):
            self.post_ln = nn.LayerNorm(out_dim, elementwise_affine=False)
            logger.info(f"MLPadapter using parameter_free_post_norm")
        if kwargs.get("no_post_norm", False):
            self.post_ln = None

        self.use_residual = kwargs.get("use_residual", False)
    def get_last_layer(self) -> torch.nn.Parameter:
        return self.mlp[self._last_linear_idx].weight

    def forward(self, x):

        if self.pre_ln is not None:
            x_before_ln = x
            x = self.pre_ln(x)
        delta = self.mlp(x)
        if self.post_ln is not None:
            return self.post_ln(delta)
        if self.use_residual:
            if self.pre_ln is not None:
                return x_before_ln + delta
            return x + delta
        return delta

class ResidualAdapter(nn.Module):
    def __init__(self, dim, zero_init=True):
        super().__init__()
        self.down = nn.Linear(dim, dim)
        self.up = nn.Linear(dim, dim)
        self.act = nn.GELU() # 加个非线性，更强

        self.pre_ln = nn.LayerNorm(dim)

        if zero_init:
            self._zero_init()
    def forward(self, x):
        x = self.pre_ln(x)
        delta = self.up(self.act(self.down(x)))
        return delta

    def _zero_init(self):
        # 初始化输出的delta为0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


class AttentionAdapter(nn.Module):
    """Attention adapter: multihead attention + FFN, no residual connection.

    Args:
        dim: Input dimension (also used as default output dimension)
        out_dim: Output dimension (defaults to dim if not specified)
        num_heads: Number of attention heads (default: 8)
        ffn_hidden_dim: Hidden dimension of FFN (defaults to 4 * dim)
        dropout: Dropout rate for attention and FFN (default: 0.0)
    """
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        in_dim = kwargs.get("in_dim", dim)
        out_dim = kwargs.get("out_dim", dim)
        num_heads = kwargs.get("num_heads", 8)
        ffn_hidden_dim = kwargs.get("ffn_hidden_dim", 4 * dim)
        dropout = kwargs.get("dropout", 0.0)

        # Optional pre-norm
        if kwargs.get("pre_norm", False):
            self.pre_ln = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.pre_ln = None

        # Multi-head attention
        # If in_dim != dim, we need a projection before attention
        self.input_proj = nn.Linear(in_dim, dim) if in_dim != dim else None
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # FFN: Linear -> GELU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, out_dim),
        )

        # Optional post-norm
        self.post_ln = None
        if kwargs.get("post_norm", False):
            self.post_ln = nn.LayerNorm(out_dim)
        if kwargs.get("parameter_free_post_norm", False):
            self.post_ln = nn.LayerNorm(out_dim, elementwise_affine=False)
            logger.info(f"AttentionAdapter using parameter_free_post_norm")
        if kwargs.get("no_post_norm", False):
            self.post_ln = None

        self._ffn_last_linear_idx = 2  # Last Linear in FFN (after GELU)

    def get_last_layer(self) -> torch.nn.Parameter:
        """Return the weight of the last FFN linear layer."""
        return self.ffn[self._ffn_last_linear_idx].weight

    def forward(self, x):
        """
        Args:
            x: (B, N, in_dim) input tensor

        Returns:
            (B, N, out_dim) output tensor
        """
        # Pre-norm
        if self.pre_ln is not None:
            x = self.pre_ln(x)

        # Project to attention dimension if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Multi-head attention (no residual)
        # For self-attention, query=key=value=x
        attn_out, _ = self.attention(x, x, x)

        # FFN (no residual)
        out = self.ffn(attn_out)

        # Post-norm
        if self.post_ln is not None:
            out = self.post_ln(out)

        return out


# 统一 registry，支持 build_recon_adapter(dim, cls_or_name, **kwargs)
RECON_ADAPTER_REGISTRY = {
    "FullRankAdapter": FullRankAdapter,
    "FullRankAdapterPreNorm": FullRankAdapterPreNorm,
    "ReconAdapter": ReconAdapter,
    "LowRankAdapter": LowRankAdapter,
    "LowRankAdapterStopGrad": LowRankAdapterStopGrad,
    "LowRankAdapterPreNorm": LowRankAdapterPreNorm,
    "LowRankAdapterNoNorm": LowRankAdapterNoNorm,
    "ResidualAdapter": ResidualAdapter,
    "MLPAdapter": MLPAdapter,
    "LinearAdapter": LinearAdapter,
    "AttentionAdapter": AttentionAdapter,
}


def build_adapter(dim, cls_or_name=None, **adapter_kwargs):
    """
    统一创建 recon adapter。cls_or_name 可以是：
    - str: 从 RECON_ADAPTER_REGISTRY 查找
    - type: 直接使用该类
    - None: 默认 FullRankAdapter
    adapter_kwargs 会传给 adapter 的 __init__（如 LowRankAdapter 的 bottleneck_dim）
    """
    if cls_or_name is None:
        cls_or_name = "FullRankAdapter"
    if isinstance(cls_or_name, type):
        return cls_or_name(dim, **adapter_kwargs)
    cls = RECON_ADAPTER_REGISTRY.get(cls_or_name, FullRankAdapter)
    return cls(dim, **adapter_kwargs)


@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name:str, num_tokens=256, hidden_size =None,**kwargs):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        zero_init = kwargs.get("zero_init", False)
        if zero_init:
            config = SiglipConfig.from_pretrained(self.model_name)
            full_model = SiglipVisionModel(config.vision_config)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)  # 不注册为子模块，保持 state_dict 兼容旧 checkpoint
            logger.warning(f"using zero_init model")
        else:
            full_model = SiglipVisionModel.from_pretrained(self.model_name)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)  # 不注册为子模块，保持 state_dict 兼容旧 checkpoint

        # 随机重初始化 patch embedding（其余预训练权重保持不变）
        reset_patch_embedding = kwargs.get("reset_patch_embedding", False)
        if reset_patch_embedding:
            pe = self.model.embeddings.patch_embedding
            nn.init.normal_(pe.weight, std=0.02)
            if pe.bias is not None:
                nn.init.zeros_(pe.bias)
            logger.info("SigLIP2wNorm: patch_embedding randomly re-initialized (std=0.02), all other weights kept pretrained")

        # 这里面是包含一个output_head
        # remove the affine of final layernorm
        # self.model.post_layernorm.elementwise_affine = False
        # # remove the param
        # self.model.post_layernorm.weight = None
        # self.model.post_layernorm.bias = None
        post_norm_type = kwargs.get("post_norm_type", "parameter_free")
        self.post_norm_type = post_norm_type

        if post_norm_type == "model_post_norm":
            logger.info(f"using model_post_norm post_norm")
            self.post_layernorm = self.model.post_layernorm
        elif post_norm_type == "parameter_free":
            logger.info(f"using parameter_free post_norm")
            self.post_layernorm = nn.LayerNorm(self.model.config.hidden_size, elementwise_affine=False)
        elif post_norm_type == "parameter_free_with_scale_shift":
            logger.info(f"using parameter_free_with_scale_shift post_norm")
            self.post_layernorm = ReconAdapter(self.model.config.hidden_size)
        elif post_norm_type == "parameter_free_with_low_rank":
            logger.info(f"using parameter_free_with_low_rank post_norm")
            self.post_layernorm = LowRankAdapter(self.model.config.hidden_size)
        elif post_norm_type == "parameter_free_with_full_rank":
            logger.info(f"using parameter_free_with_full_rank post_norm")
            self.post_layernorm = FullRankAdapter(self.model.config.hidden_size)
        elif post_norm_type == "parameter_free_with_low_rank_pre_norm":
            logger.info(f"using parameter_free_with_low_rank_pre_norm post_norm")
            self.post_layernorm = LowRankAdapterPreNorm(self.model.config.hidden_size)
        elif post_norm_type == "parameter_free_with_low_rank_no_norm":
            logger.info(f"using parameter_free_with_low_rank_no_norm post_norm")
            self.post_layernorm = LowRankAdapterNoNorm(self.model.config.hidden_size)
        elif post_norm_type == "parameter_free_with_low_rank_stop_grad":
            logger.info(f"using parameter_free_with_low_rank_stop_grad post_norm")
            self.post_layernorm = LowRankAdapterStopGrad(self.model.config.hidden_size)
        elif post_norm_type == "no_norm":
            self.post_layernorm = None
            logger.info(f"using no_norm post_norm")
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")
        if hidden_size:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        self.output_head_type = kwargs.get("output_head_type", "siglip_MAP")

        self.select_layer_idx = kwargs.get("select_layer_idx", -1)

        need_proj = kwargs.get("need_proj", False)
        if need_proj:
            adapter_kwargs = kwargs.get("adapter_kwargs", {})
            adapter_cls = kwargs.get("adapter_cls", {})
            self.proj = build_adapter(
                self.model.config.hidden_size, adapter_cls, **adapter_kwargs
            )
        # Optional: enable gradient checkpointing at init (e.g. via encoder_params.gradient_checkpointing)
        if kwargs.get("gradient_checkpointing", False):
            self.set_gradient_checkpointing()

    def set_gradient_checkpointing(self):
        """Enable gradient checkpointing on the SigLIP vision model to save memory.
        使用 SiglipVisionModel (PreTrainedModel) 的标准接口 gradient_checkpointing_enable()。
        """
        enabled = False
        full_model = getattr(self, "_full_vision_model", None)
        if full_model is not None and hasattr(full_model, "gradient_checkpointing_enable"):
            full_model.gradient_checkpointing_enable()
            logger.info("SigLIP2wNorm: gradient checkpointing enabled (via SiglipVisionModel)")
            enabled = True
        # Fallback: 手动设置 gradient_checkpointing 和 _gradient_checkpointing_func
        if not enabled:
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            checkpoint_fn = partial(checkpoint, use_reentrant=False)
            def _enable(module):
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    if getattr(module, "_gradient_checkpointing_func", None) is None:
                        module._gradient_checkpointing_func = checkpoint_fn
            self.model.apply(_enable)
            logger.info("SigLIP2wNorm: gradient checkpointing enabled (fallback)")
            enabled = True
        if not enabled:
            logger.warning("SigLIP2wNorm: gradient checkpointing not available")
    def set_training_mode(self, mode: str = 'full', return_stats: bool = False):
        """
        设置 encoder 的训练模式，控制哪些参数可训练。
        """
        self.requires_grad_(False)
        if mode == 'full':
            logger.info("training full model")
            self.requires_grad_(True)
        elif mode == "no_sigliphead":
            logger.info("siglip head no grad, siglip postnorm no grad")
            self.requires_grad_(True)
            for param in self.model.head.parameters():
                param.requires_grad = False
                logger.debug(f"siglip head {param.shape} requires_grad: {param.requires_grad}")
            for param in self.model.post_layernorm.parameters():
                param.requires_grad = False
                logger.debug(f"siglip postnorm {param.shape} requires_grad: {param.requires_grad}")
        elif mode == "patchemb":
            logger.info("training patch embedding")
            if hasattr(self.model, 'embeddings') and self.model.embeddings is not None:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = True
                    logger.debug(f"patch embedding {param.shape} requires_grad: {param.requires_grad}")
        elif mode == 'frozen':
            self.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        if return_stats:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            # 统计self.model之外的参数
            model_params = sum(p.numel() for p in self.model.parameters())
            other_params = trainable_params - model_params


            return {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'model_params': model_params,
                'other_params': other_params,
            }
        return None
    @torch.no_grad()
    def get_image_features(self, images):
        return self.model(images).pooler_output
    def output_head_forward(self, hidden_states, mean_first = True):
        assert self.model.head is not None, "output_head is not initialized"

        # hidden_states: (B, N, H, W) or (B, N, C), 是已经没有prefix tokens的输出
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
            logger.debug(f"hidden_states shape: {hidden_states.shape}")

        if self.output_head_type == "siglip_MAP":
            hidden_states = self.model.head(hidden_states)
        elif self.output_head_type == "siglip_MAP_with_postnorm":
            assert self.model.post_layernorm.elementwise_affine == True, "post_layernorm should be with affine"
            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))
        elif self.output_head_type == "siglip_MAP_with_postnorm_multiscale":
            b, n, c = hidden_states.shape
            h = w = int(sqrt(n))
            hidden_states_2d = hidden_states.transpose(1, 2).view(b, c, h//2, 2, w//2, 2)
            hidden_states_scale_2 = hidden_states_2d.permute(0, 3, 5, 2, 4, 1).reshape(b*4, h//2 * w//2, c)
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")
            hidden_states_scale_2 = self.model.head(self.model.post_layernorm(hidden_states_scale_2))
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")

            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))

            hidden_states = torch.cat([hidden_states, hidden_states_scale_2], dim=0)

            logger.debug(f"hidden_states shape: {hidden_states.shape}")



        else:
            raise ValueError(f"Unknown output_head_type: {self.output_head_type}")
        logger.debug(f"hidden_states shape after output_head: {hidden_states.shape}")
        return hidden_states

    def get_last_layer(self):
        return self.model.encoder.layers[-1].mlp.fc2.weight

    def forward(self, images,**kwargs):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """

        # 获取 embeddings
        hidden_states = self.model.embeddings(images, interpolate_pos_encoding=True)

        logger.debug(self.select_layer_idx)
        if self.select_layer_idx == -1:
            # 跑完所有层，取 last_hidden_state
            encoder_outputs = self.model.encoder(inputs_embeds=hidden_states)
            pre_postnorm_hidden_states = encoder_outputs.last_hidden_state
        elif self.select_layer_idx == 0:
            # index 0 = patch embedding 输出，一层 transformer 都不跑
            pre_postnorm_hidden_states = hidden_states
        else:
            # index k（1~12）= 跑前 k 个 transformer block 后停止
            # 对应 block 0 ~ block k-1 的输出
            cur = hidden_states
            for i, layer in enumerate(self.model.encoder.layers):
                cur = layer(cur, attention_mask=None)
                if i + 1 == self.select_layer_idx:
                    break
            pre_postnorm_hidden_states = cur

        logger.debug(f"pre_postnorm_hidden_states max: {pre_postnorm_hidden_states.max()}, min: {pre_postnorm_hidden_states.min()}, mean: {pre_postnorm_hidden_states.mean()}")

        if hasattr(self, "proj"):
            pre_postnorm_hidden_states = self.proj(pre_postnorm_hidden_states)
               # 如果需要 postnorm 之后的结果
        if self.post_layernorm:
            post_postnorm_hidden_states = self.post_layernorm(pre_postnorm_hidden_states) #给decoder用的，无参数的layernorm
        # pooler_output = self.model.head(post_postnorm_hidden_states) if self.model.use_head else None

            image_features = post_postnorm_hidden_states #[B N D]
        else:
            logger.debug("no norm")
            image_features = pre_postnorm_hidden_states
        return_aux = kwargs.get("return_aux", False)
        if return_aux:
            aux = {}
            aux["hidden_states_before_postnorm"] = pre_postnorm_hidden_states

            return image_features, aux
        return image_features


@register_encoder()
class SigLIP2wResidual(nn.Module):
    def __init__(self, model_name:str, num_tokens=256, **kwargs):
        super().__init__()
        logger.info(f"SigLIP2wResidual init")
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(self.model_name).vision_model
        # 这里面是包含一个output_head
        # remove the affine of final layernorm
        # self.model.post_layernorm.elementwise_affine = False
        # # remove the param
        # self.model.post_layernorm.weight = None
        # self.model.post_layernorm.bias = None
        post_norm_type = kwargs.get("post_norm_type", "parameter_free")
        self.post_norm_type = post_norm_type
        if post_norm_type == "parameter_free":
            logger.info(f"using parameter_free post_norm")
            self.post_layernorm = nn.LayerNorm(self.model.config.hidden_size, elementwise_affine=False)
        elif post_norm_type == "model_post_norm":
            logger.info(f"using model_post_norm post_norm")
            self.post_layernorm = self.model.post_layernorm
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.residual_adapter = ResidualAdapter(self.model.config.hidden_size)
        self.output_head_type = kwargs.get("output_head_type", "siglip_MAP")

        self.residual_stop_sementic = kwargs.get("residual_stop_sementic", False)
        logger.info(f"residual_stop_sementic: {self.residual_stop_sementic}, 如果为true，表示给recon分支的，将带上stop gradient的语义信息")
    def set_training_mode(self, mode: str = 'full', return_stats: bool = False):
        """
        设置 encoder 的训练模式，控制哪些参数可训练。
        """
        self.requires_grad_(False)
        if mode == 'full':
            logger.info("training full model")
            self.requires_grad_(True)
        elif mode == "patchemb":
            logger.info("training patch embedding")
            if hasattr(self.model, 'embeddings') and self.model.embeddings is not None:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = True
                    logger.debug(f"patch embedding {param.shape} requires_grad: {param.requires_grad}")
        elif mode == 'frozen':
            self.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        if return_stats:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            # 统计self.model之外的参数
            model_params = sum(p.numel() for p in self.model.parameters())
            other_params = trainable_params - model_params


            return {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'model_params': model_params,
                'other_params': other_params,
            }
        return None
    @torch.no_grad()
    def get_image_features(self, images):
        _, aux = self.forward(images, return_aux=True)
        hidden_states_before_postnorm = aux["hidden_states_before_postnorm"]
        return self.model.head(self.model.post_layernorm(hidden_states_before_postnorm))

    def output_head_forward(self, hidden_states, mean_first = True):
        assert self.model.head is not None, "output_head is not initialized"

        # hidden_states: (B, N, H, W) or (B, N, C), 是已经没有prefix tokens的输出
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
            logger.debug(f"hidden_states shape: {hidden_states.shape}")

        if self.output_head_type == "siglip_MAP":
            hidden_states = self.model.head(hidden_states)
        elif self.output_head_type == "siglip_MAP_with_postnorm":
            assert self.model.post_layernorm.elementwise_affine == True, "post_layernorm should be with affine"
            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))
        elif self.output_head_type == "siglip_MAP_with_postnorm_multiscale":
            b, n, c = hidden_states.shape
            h = w = int(sqrt(n))
            hidden_states_2d = hidden_states.transpose(1, 2).view(b, c, h//2, 2, w//2, 2)
            hidden_states_scale_2 = hidden_states_2d.permute(0, 3, 5, 2, 4, 1).reshape(b*4, h//2 * w//2, c)
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")
            hidden_states_scale_2 = self.model.head(self.model.post_layernorm(hidden_states_scale_2))
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")

            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))

            hidden_states = torch.cat([hidden_states, hidden_states_scale_2], dim=0)

            logger.debug(f"hidden_states shape: {hidden_states.shape}")



        else:
            raise ValueError(f"Unknown output_head_type: {self.output_head_type}")
        logger.debug(f"hidden_states shape after output_head: {hidden_states.shape}")
        return hidden_states

    def get_last_layer(self):
        return self.model.encoder.layers[-1].mlp.fc2.weight

    def forward(self, images,**kwargs):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """

        # 获取 embeddings
        hidden_states = self.model.embeddings(images, interpolate_pos_encoding=True)

        # 直接调用 encoder，获取 postnorm 之前的 hidden states
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )
        pre_postnorm_hidden_states = encoder_outputs.last_hidden_state

        residual = self.residual_adapter(pre_postnorm_hidden_states)
        # 如果需要 postnorm 之后的结果
        if self.residual_stop_sementic:
            post_postnorm_hidden_states = self.post_layernorm(residual + (pre_postnorm_hidden_states - residual).detach())
        else:
            post_postnorm_hidden_states = self.post_layernorm(residual) #给decoder用的，无参数的layernorm
        # pooler_output = self.model.head(post_postnorm_hidden_states) if self.model.use_head else None

        image_features = post_postnorm_hidden_states #[B N D]
        return_aux = kwargs.get("return_aux", False)
        if return_aux:
            aux = {}
            aux["hidden_states_before_postnorm"] = pre_postnorm_hidden_states - residual #含义就是在过output head之前的token
            aux["sementic_tokens"] = self.model.post_layernorm(pre_postnorm_hidden_states - residual) # 用过layernorm的
            aux["recon_tokens"] = image_features # 用过layernorm的
            return image_features, aux
        return image_features


@register_encoder()
class SigLIP2wtwohead(nn.Module):
    def __init__(self, model_name:str, num_tokens=256, **kwargs):
        super().__init__()
        logger.info(f"SigLIP2wtwohead init")
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(self.model_name).vision_model
        # 这里面是包含一个output_head
        # remove the affine of final layernorm
        # self.model.post_layernorm.elementwise_affine = False
        # # remove the param
        # self.model.post_layernorm.weight = None
        # self.model.post_layernorm.bias = None

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.sementic_head_type = kwargs.get("sementic_head_type", "siglip_MAP")
        self.recon_head_type = kwargs.get("recon_head_type", "parameter_free_postnorm")
        if self.recon_head_type == "recon_adapter":
            self.recon_adapter = build_adapter(
                self.model.config.hidden_size,
                cls_or_name=kwargs.get("recon_adapter_cls", "FullRankAdapter"),
                **kwargs.get("recon_adapter_kwargs", {}),
            )

        elif self.recon_head_type == "parameter_free_postnorm":
            self.recon_post_layernorm = nn.LayerNorm(self.model.config.hidden_size, elementwise_affine=False)

        if self.sementic_head_type == "sementic_adapter" or self.sementic_head_type == "sementic_adapter_w_postnorm":
            self.sementic_adapter = build_adapter(
                self.model.config.hidden_size,
                cls_or_name=kwargs.get("sementic_adapter_cls", None),
                **kwargs.get("sementic_adapter_kwargs", {}),
            )

        # 用哪种方式来获取clip要用的image features
        self.get_image_features_type = kwargs.get("get_image_features_type", "hidden_states_before_postnorm")
        self.output_head_type = kwargs.get("output_head_type", "siglip_MAP")


    def set_training_mode(self, mode: str = 'full', return_stats: bool = False):
        """
        设置 encoder 的训练模式，控制哪些参数可训练。
        """
        self.requires_grad_(False)
        if mode == 'full':
            logger.info("training full model")
            self.requires_grad_(True)
        elif mode == "recon_adapter_only":
            logger.info("training recon adapter only")
            for param in self.recon_adapter.parameters():
                param.requires_grad = True
                logger.debug(f"recon adapter {param.shape} requires_grad: {param.requires_grad}")

        elif mode == "no_sigliphead":
            logger.info("siglip head no grad, siglip postnorm no grad")
            self.requires_grad_(True)
            for param in self.model.head.parameters():
                param.requires_grad = False
                logger.debug(f"siglip head {param.shape} requires_grad: {param.requires_grad}")
            for param in self.model.post_layernorm.parameters():
                param.requires_grad = False
                logger.debug(f"siglip postnorm {param.shape} requires_grad: {param.requires_grad}")
        elif mode == "patchemb":
            logger.info("training patch embedding")
            if hasattr(self.model, 'embeddings') and self.model.embeddings is not None:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = True
                    logger.debug(f"patch embedding {param.shape} requires_grad: {param.requires_grad}")
        elif mode == 'frozen':
            self.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        if return_stats:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            # 统计self.model之外的参数
            model_params = sum(p.numel() for p in self.model.parameters())
            other_params = trainable_params - model_params


            return {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'model_params': model_params,
                'other_params': other_params,
            }
        return None
    @torch.no_grad()
    def get_image_features(self, images):
        _, aux = self.forward(images, return_aux=True)
        if self.get_image_features_type == "sementic_adapter_w_postnorm":
            # 在semetic token上相当于对齐了postnorm
            hidden_states = aux["sementic_tokens"]
            return self.model.head(hidden_states)
        elif self.get_image_features_type == "hidden_states_before_postnorm":
            hidden_states_before_postnorm = aux["hidden_states_before_postnorm"]
            return self.model.head(self.model.post_layernorm(hidden_states_before_postnorm))

        elif self.get_image_features_type == "sementic_tokens":
            sementic_hidden_states = aux["sementic_tokens"]
            return self.model.head(sementic_hidden_states)
        else:
            raise ValueError(f"Unknown get_image_features_type: {self.get_image_features_type}")

    def output_head_forward(self, hidden_states, mean_first = True):
        assert self.model.head is not None, "output_head is not initialized"

        # hidden_states: (B, N, H, W) or (B, N, C), 是已经没有prefix tokens的输出
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
            logger.debug(f"hidden_states shape: {hidden_states.shape}")

        if self.output_head_type == "siglip_MAP":
            hidden_states = self.model.head(hidden_states)
        elif self.output_head_type == "siglip_MAP_with_postnorm":
            assert self.model.post_layernorm.elementwise_affine == True, "post_layernorm should be with affine"
            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))
        elif self.output_head_type == "siglip_MAP_with_postnorm_multiscale":
            b, n, c = hidden_states.shape
            h = w = int(sqrt(n))
            hidden_states_2d = hidden_states.transpose(1, 2).view(b, c, h//2, 2, w//2, 2)
            hidden_states_scale_2 = hidden_states_2d.permute(0, 3, 5, 2, 4, 1).reshape(b*4, h//2 * w//2, c)
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")
            hidden_states_scale_2 = self.model.head(self.model.post_layernorm(hidden_states_scale_2))
            logger.debug(f"hidden_states_scale_2 shape: {hidden_states_scale_2.shape}")

            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))

            hidden_states = torch.cat([hidden_states, hidden_states_scale_2], dim=0)

            logger.debug(f"hidden_states shape: {hidden_states.shape}")



        else:
            raise ValueError(f"Unknown output_head_type: {self.output_head_type}")
        logger.debug(f"hidden_states shape after output_head: {hidden_states.shape}")
        return hidden_states

    def get_last_layer(self):
        return self.model.encoder.layers[-1].mlp.fc2.weight

    def get_sementic_features(self, hidden_states):
        # hidden_states: [B N C] before postnorm
        if self.sementic_head_type == "siglip_MAP":
            b, n, c = hidden_states.shape
            hidden_states = hidden_states.reshape(b*n, 1, c)
            hidden_states = self.model.head(self.model.post_layernorm(hidden_states))
            return hidden_states.reshape(b, n, c)
        elif self.sementic_head_type == "siglip_MAP_pooler":
            return self.model.head(self.model.post_layernorm(hidden_states)) #[B C]

        elif self.sementic_head_type == "sementic_adapter":
            logger.debug("using sementic_adapter")
            sementic_tokens = self.sementic_adapter(hidden_states)
            # if self.sementic_head_need_postnorm:
            #     sementic_tokens = self.model.post_layernorm(sementic_tokens)
            return sementic_tokens
        elif self.sementic_head_type == "sementic_adapter_w_postnorm":
            logger.debug("using sementic_adapter_w_postnorm")
            sementic_tokens = self.sementic_adapter(hidden_states)
            return self.model.post_layernorm(sementic_tokens)
        elif self.sementic_head_type == "postnorm":
            return self.model.post_layernorm(hidden_states)
        else:
            raise ValueError(f"Unknown sementic_head_type: {self.sementic_head_type}")

    def get_recon_features(self, hidden_states):
        # hidden_states: [B N C] before postnorm
        if self.recon_head_type == "recon_adapter":
            logger.debug("using recon_adapter")
            return self.recon_adapter(hidden_states)
        elif self.recon_head_type == "parameter_free_postnorm":
            return self.recon_post_layernorm(hidden_states)
        else:
            raise ValueError(f"Unknown recon_head_type: {self.recon_head_type}")
    def forward(self, images,**kwargs):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """

        # 获取 embeddings
        hidden_states = self.model.embeddings(images, interpolate_pos_encoding=True)

        # 直接调用 encoder，获取 postnorm 之前的 hidden states
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )
        pre_postnorm_hidden_states = encoder_outputs.last_hidden_state

        sementic_tokens = self.get_sementic_features(pre_postnorm_hidden_states)
        recon_tokens = self.get_recon_features(pre_postnorm_hidden_states)
        logger.debug(f"sementic_tokens shape: {sementic_tokens.shape}, recon_tokens shape: {recon_tokens.shape}")
        image_features = recon_tokens
        return_aux = kwargs.get("return_aux", False)
        if return_aux:
            aux = {}
            aux["hidden_states_before_postnorm"] = pre_postnorm_hidden_states
            aux["sementic_tokens"] = sementic_tokens # 用过layernorm的
            aux["recon_tokens"] = recon_tokens # 用过layernorm的
            return image_features, aux

        return image_features


class TokenMerger(nn.Module):
    """Merge tokens from two backbones. Supports concat+linear, add, learned_weight."""

    def __init__(self, dim: int, merger_type: str = "concat", **kwargs):
        super().__init__()
        self.merger_type = merger_type

        # Optional pre-projection for recon tokens before concat (e.g. 768->256)
        recon_pre_proj_kwargs = kwargs.get("recon_pre_proj_kwargs", None)
        if recon_pre_proj_kwargs is not None:
            recon_pre_proj_cls = kwargs.get("recon_pre_proj_cls", "LinearAdapter")
            kw = dict(recon_pre_proj_kwargs)
            pre_proj_dim = kw.pop("dim", dim)
            self.recon_pre_proj = build_adapter(pre_proj_dim, cls_or_name=recon_pre_proj_cls, **kw)
            logger.info(f"TokenMerger: recon_pre_proj enabled, cls={recon_pre_proj_cls}, kwargs={recon_pre_proj_kwargs}")
        else:
            self.recon_pre_proj = None

        self.backbone = MLPAdapter(**kwargs.get("backbone_kwargs", {}))

        def _proj_kwargs(key):
            kw = dict(kwargs.get(key, {}))
            if "dim" not in kw:
                kw["dim"] = dim
            return kw
        no_sementic_proj = kwargs.get("no_sementic_proj", False)
        if no_sementic_proj:
            self.sementic_proj = None
            sementic_proj_cls = None
            logger.warning("No sementic proj, sementic tokens will be merged tokens directly")
        else:
            sementic_proj_cls = kwargs.get("sementic_proj_cls", "LinearAdapter")
            sementic_kw = _proj_kwargs("sementic_proj_kwargs")
            self.sementic_proj = build_adapter(sementic_kw.pop("dim", dim), cls_or_name=sementic_proj_cls, **sementic_kw)

        no_recon_proj = kwargs.get("no_recon_proj", False)
        if no_recon_proj:
            self.recon_proj = None
            recon_proj_cls = None
            logger.warning("No recon proj, recon tokens will be merged tokens directly")
        else:
            recon_proj_cls = kwargs.get("recon_proj_cls", "LinearAdapter")
            recon_kw = _proj_kwargs("recon_proj_kwargs")
            self.recon_proj = build_adapter(recon_kw.pop("dim", dim), cls_or_name=recon_proj_cls, **recon_kw)

        # Dropout applied only to recon branch during training, before backbone.
        # This mask is shared across the merged tokens so that sementic branch
        # sees the clean features while recon branch can optionally see a dropped version.
        self.recon_dropout_p = float(kwargs.get("recon_dropout_p", 0.0))
        if self.recon_dropout_p < 0.0 or self.recon_dropout_p > 1.0:
            raise ValueError(f"recon_dropout_p must be in [0, 1], got {self.recon_dropout_p}")

        # Token-level or instance-level recon dropout with batch mean replacement.
        # Forces decoder to use both semantic and recon branches.
        self.recon_token_drop_rate = float(kwargs.get("recon_token_drop_rate", 0.0))
        self.recon_token_drop_level = kwargs.get("recon_token_drop_level", "token")  # "token" or "instance"
        if self.recon_token_drop_rate > 0:
            logger.info(f"Recon token dropout enabled: rate={self.recon_token_drop_rate}, level={self.recon_token_drop_level}")

        self.noise_before_recon = kwargs.get("noise_before_recon", False)
        if self.noise_before_recon:

            self.noise_tau = kwargs.get("noise_tau", 0.8)
            logger.warning(f"Noise before recon is enabled, noise_tau={self.noise_tau} 记得MLPadapter一定要加postnorm")

            self.noise_normalization = kwargs.get("noise_normalization", False)
            if self.noise_normalization and kwargs.get("norm_stat_path", None) is not None:
                stats = torch.load(kwargs.get("norm_stat_path"), map_location='cpu')
                self.latent_mean = stats.get('mean', None)
                self.latent_var = stats.get('var', None)

                self.eps = kwargs.get("eps", 1e-6)
                logger.debug(f"latent mean mean: {self.latent_mean.mean().item()}, latent mean std: {self.latent_mean.std().item()}")
                logger.debug(f"latent var mean: {self.latent_var.mean().item()}, latent var std: {self.latent_var.std().item()}")
                logger.warning(f"Noise normalization is enabled, latent_mean={self.latent_mean}, latent_var={self.latent_var}, eps={self.eps}")
            else:
                logger.warning("Noise normalization is disabled")
        self.merge_linear_interpolation = kwargs.get("merge_linear_interpolation", False)
        if self.merge_linear_interpolation:
            self.semetic_norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.merge_distribution = kwargs.get("merge_distribution", None)

        self.force_drop = kwargs.get("force_drop", False)
        if self.force_drop:
            logger.warning("force drop here")

        assert not self.merge_distribution or not self.merge_linear_interpolation, "Only one of merge_distribution or merge_linear_interpolation can be enabled"

        logger.info(f"TokenMerger init: merger_type={merger_type}, sementic_proj_cls={sementic_proj_cls}, recon_proj_cls={recon_proj_cls}")
        logger.info(f"TokenMerger init: backbone_kwargs={kwargs.get('backbone_kwargs', {})}, sementic_proj_kwargs={kwargs.get('sementic_proj_kwargs', {})}, recon_proj_kwargs={kwargs.get('recon_proj_kwargs', {})}")
    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("TokenMerger: training full model")
            self.requires_grad_(True)
        elif mode == "recon_proj_only":
            logger.info("TokenMerger: training recon proj only")
            for p in self.recon_proj.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_normalization:
            latent_mean = self.latent_mean.to(x.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(x.device) if self.latent_var is not None else 1
            x = (x - latent_mean) / torch.sqrt(latent_var + self.eps)
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        x = x + noise
        if self.noise_normalization:
            x = x * torch.sqrt(latent_var + self.eps) + latent_mean
        return x

    def get_last_layer(self) -> torch.nn.Parameter:
        return self.backbone.get_last_layer()

    def linear_interpolation_random(self, token1, token2, distribution='uniform', alpha=1.0):
        """
        对两个 Token 进行基于向量相加的随机线性插值。
        每个 batch 样本使用独立的 t，以增强 Mixup 正则化效果。

        参数:
            token1: 第一个 token (Tensor, 首维为 batch)
            token2: 第二个 token (Tensor, 首维为 batch)
            distribution: 'uniform' (均匀分布) 或 'beta' (Mixup常用)
            alpha: Beta 分布的参数 (仅在 distribution='beta' 时有效)
        """
        batch_size = token1.size(0)
        device = token1.device

        # 1. 按 batch 随机采样 t (每个样本独立)
        if distribution == 'uniform':
            t = torch.rand(batch_size, device=device)
        elif distribution == 'beta':
            t = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
        else:
            raise ValueError("Unknown distribution")

        # 2. reshape 以广播到 (B, ..., D): t -> (B, 1, 1, ...)
        t = t.view(batch_size, *([1] * (token1.dim() - 1)))

        # 3. 线性插值: (1-t)*A + t*B
        mixed_token = (1 - t) * token1 + t * token2
        return mixed_token

    def distribution_interpolation(self, token1, token2, distribution='uniform', **kwargs):
        """
        对两个 Token 进行基于向量相加的随机线性插值。
        每个 batch 样本使用独立的 t，以增强 Mixup 正则化效果。

        参数:
            token1: 第一个 token (Tensor, 首维为 batch). origin point
            token2: 第二个 token (Tensor, 首维为 batch).
            distribution: 'uniform' (均匀分布) 或 'beta' (Mixup常用)
            alpha: Beta 分布的参数 (仅在 distribution='beta' 时有效)
        """
        batch_size = token1.size(0)
        n_tokens = token1.size(1)
        device = token1.device

        # 1. 按 batch 随机采样 t (每个样本独立)
        if distribution == 'uniform':
            t = torch.rand(batch_size, device=device)
            logger.debug(f"Uniform distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == "token_uniform":
            t = torch.rand((batch_size, n_tokens), device=device)
            logger.debug(f"Token uniform distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == "bool":
            t = torch.randint(0, 2, (batch_size,), device=device).float()  # float 以支持插值及 mean/std
            logger.debug(f"Bool distribution t: {t.mean().item()}, {t.std().item()}")
        else:
            raise ValueError("Unknown distribution")

        # 2. reshape 以广播到 (B, ..., D): t -> (B, ..1)
        t = t.view(*t.shape, *([1] * (token1.dim() - t.dim())))
        logger.debug(f"t shape: {t.shape}")
        # 3. 线性插值: (1-t)*A + t*B
        mixed_token = (1 - t) * token1 + t * token2
        return mixed_token, t

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, return_distribution: bool = False) -> torch.Tensor:
        # Recon token dropout: replace with batch mean to force decoder to use both branches.
        if (self.training or self.force_drop) and self.recon_token_drop_rate > 0:
            B, N, D = recon_tokens.shape
            recon_mean = recon_tokens.mean(dim=0, keepdim=True)  # [1, N, D]
            if self.recon_token_drop_level == "token":
                mask = torch.rand(B, N, 1, device=recon_tokens.device, dtype=recon_tokens.dtype) > self.recon_token_drop_rate
            elif self.recon_token_drop_level == "instance":
                mask = torch.rand(B, 1, 1, device=recon_tokens.device, dtype=recon_tokens.dtype) > self.recon_token_drop_rate
            else:
                raise ValueError(f"Unknown recon_token_drop_level: {self.recon_token_drop_level}")
            logger.debug(f"dropout {mask} as {recon_mean}")
            recon_tokens = torch.where(mask, recon_tokens, recon_mean.expand_as(recon_tokens))

        # Optional pre-projection for recon tokens before concat
        if self.recon_pre_proj is not None:
            recon_tokens = self.recon_pre_proj(recon_tokens)
            logger.debug(f"recon pre proj:{recon_tokens.shape}")
        if self.merger_type == "concat":
            hidden_states = torch.cat([sementic_tokens, recon_tokens], dim=-1)
        else:
            raise ValueError(f"Unknown merger_type: {self.merger_type}")

        # Always keep a clean version for sementic branch.
        hidden_states = self.backbone(hidden_states)

        # Optionally apply dropout to recon branch only during training.
        if self.training and self.recon_dropout_p > 0.0:
            # Drop on the recon part before backbone so that only recon branch
            # is affected after projection.
            if self.merger_type != "concat":
                raise ValueError("recon_dropout_p currently only supports merger_type='concat'")
            # hidden_states shape: (B, T, D_concat), where last dim is [sementic, recon]

            dropout_mask = torch.empty_like(recon_tokens).bernoulli_(1.0 - self.recon_dropout_p) / (1.0 - self.recon_dropout_p)
            recon_part_dropped = recon_tokens * dropout_mask
            recon_hidden_states = torch.cat([sementic_tokens, recon_part_dropped], dim=-1)
            recon_hidden_states = self.backbone(recon_hidden_states)
            logger.debug(f"recon_hidden_states norm: {recon_hidden_states.norm(dim=-1).mean().item()}, hidden_states norm: {hidden_states.norm(dim=-1).mean().item()}")
            logger.debug(f"recon_hidden_states std: {recon_hidden_states.std().item()}, hidden_states std: {hidden_states.std().item()}")
        else:
            recon_hidden_states = hidden_states
        # print("hidden_states dtype:", hidden_states.dtype)
        # print(self.training, self.merge_linear_interpolation, self.merge_distribution)
        # logger.debug(f"hidden_states norm: {hidden_states.norm(dim=-1).mean().item()}, sementic_tokens norm: {sementic_tokens.norm(dim=-1).mean().item()} recon_tokens norm: {recon_tokens.norm(dim=-1).mean().item()}")
        # logger.debug(f"hidden_states std: {hidden_states.std().item()}, sementic_tokens std: {sementic_tokens.std().item()} recon_tokens std: {recon_tokens.std().item()}")
        if self.training and self.merge_linear_interpolation:
            logger.debug(f"Linear interpolation")
            # 实验版
            hidden_states = self.linear_interpolation_random(self.semetic_norm(sementic_tokens), hidden_states)
            # logger.debug(f"Before norm, hidden_states stats: mean={hidden_states.mean().item()}, std={hidden_states.std().item()}, min={hidden_states.min().item()}, max={hidden_states.max().item()}")
            hidden_states = self.semetic_norm(hidden_states)
        elif self.training and self.merge_distribution:
            logger.debug(f"Distribution interpolation")
            hidden_states, t = self.distribution_interpolation(sementic_tokens, hidden_states, distribution=self.merge_distribution)
        elif self.merge_distribution:
            t = None

        if self.sementic_proj is not None:
            sementic_hidden_states = self.sementic_proj(hidden_states)
        else:
            sementic_hidden_states = hidden_states

        if self.training and self.noise_before_recon:
            logger.debug(f"Noising hidden_states before recon")
            recon_hidden_states = self.noising(recon_hidden_states)


        if self.recon_proj is not None:
            recon_hidden_states = self.recon_proj(recon_hidden_states)


            # logger.debug(f"After norm, hidden_states stats: mean={hidden_states.mean().item()}, std={hidden_states.std().item()}, min={hidden_states.min().item()}, max={hidden_states.max().item()}")
        # elif self.merger_type == "add":
        #     hidden_states =  (tokens_a + tokens_b) / 2.0
        # elif self.merger_type == "learned_weight":
        #     alpha = torch.sigmoid(self.alpha)
        #     hidden_states =  alpha * tokens_a + (1 - alpha) * tokens_b
        if return_distribution:
            return sementic_hidden_states, recon_hidden_states, hidden_states, t
        else:
            return sementic_hidden_states, recon_hidden_states, hidden_states

class ConcatMerger(nn.Module):
    """Merge tokens from two backbones via proj + LN + concat.

    recon tokens are projected to recon_proj_dim, then both branches are
    LayerNormed and concatenated along the feature dimension.
    Output dim = dim + recon_proj_dim.
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.recon_proj_dim = int(kwargs.get("recon_proj_dim", dim))
        self.output_dim = dim + self.recon_proj_dim

        # Recon projection: backbone hidden_size -> recon_proj_dim
        recon_proj_cls = kwargs.get("recon_proj_cls", "LinearAdapter")
        recon_proj_kwargs = dict(kwargs.get("recon_proj_kwargs", {}))
        if "in_dim" not in recon_proj_kwargs:
            recon_proj_kwargs["in_dim"] = dim
        if "out_dim" not in recon_proj_kwargs:
            recon_proj_kwargs["out_dim"] = self.recon_proj_dim
        self.recon_proj = build_adapter(dim, cls_or_name=recon_proj_cls, **recon_proj_kwargs)

        self.sem_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.recon_ln = nn.LayerNorm(self.recon_proj_dim, elementwise_affine=False)

        # "merged" (default): recon_hidden_states = merged 1024d (backward compatible)
        # "recon_only": recon_hidden_states = rec (recon_proj_dim d, e.g. 256d)
        self.recon_output_mode = kwargs.get("recon_output_mode", "merged")

        # Token-level dropout for recon tokens (before projection).
        # Dropped tokens are replaced with batch mean so decoder cannot
        # rely on exact recon values at every position.
        self.recon_token_drop_rate = float(kwargs.get("recon_token_drop_rate", 0.0))
        self.recon_token_drop_level = kwargs.get("recon_token_drop_level", "token")  # "token" or "instance"
        if self.recon_token_drop_rate > 0:
            logger.info(
                f"ConcatMerger: recon token dropout enabled: "
                f"rate={self.recon_token_drop_rate}, level={self.recon_token_drop_level}"
            )

        logger.info(
            f"ConcatMerger: dim={dim}, recon_proj_dim={self.recon_proj_dim}, "
            f"output_dim={self.output_dim}, recon_proj_cls={recon_proj_cls}"
        )

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("ConcatMerger: training full model")
            self.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def get_recon_last_layer(self) -> nn.Parameter:
        """Return the last trainable parameter of the recon projection."""
        if hasattr(self.recon_proj, 'get_last_layer'):
            return self.recon_proj.get_last_layer()
        last_linear = None
        for m in self.recon_proj.modules():
            if isinstance(m, nn.Linear):
                last_linear = m.weight
        return last_linear

    def sementic_proj(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Extract semantic component from merged tokens (first dim dimensions)."""
        return merged_tokens[..., :self.output_dim - self.recon_proj_dim]

    def extract_recon(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Extract recon component from merged tokens (last recon_proj_dim dimensions)."""
        return merged_tokens[..., -self.recon_proj_dim:]

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, **kwargs) -> tuple:
        # Token-level dropout on recon tokens before projection
        if self.training and self.recon_token_drop_rate > 0:
            B, N, D = recon_tokens.shape
            recon_mean = recon_tokens.mean(dim=0, keepdim=True)  # [1, N, D]
            if self.recon_token_drop_level == "token":
                mask = torch.rand(B, N, 1, device=recon_tokens.device, dtype=recon_tokens.dtype) > self.recon_token_drop_rate
            elif self.recon_token_drop_level == "instance":
                mask = torch.rand(B, 1, 1, device=recon_tokens.device, dtype=recon_tokens.dtype) > self.recon_token_drop_rate
            else:
                raise ValueError(f"Unknown recon_token_drop_level: {self.recon_token_drop_level}")
            recon_tokens = torch.where(mask, recon_tokens, recon_mean.expand_as(recon_tokens))

        recon_tokens = self.recon_proj(recon_tokens)
        sem = self.sem_ln(sementic_tokens)
        rec = self.recon_ln(recon_tokens)
        merged = torch.cat([sem, rec], dim=-1)
        recon_out = rec if self.recon_output_mode == "recon_only" else merged
        return merged, recon_out, merged


class ConcatMergerVAE(nn.Module):
    """Merge tokens via concat, with VAE reparameterization on recon branch.

    Flow:
        sem  -> LN -> [768d]
        recon -> mu_head(768->recon_proj_dim) + logvar_head(768->recon_proj_dim)
              -> reparameterize -> [recon_proj_dim d]
        merged = concat([sem_ln, z]) -> [768 + recon_proj_dim]d

    KL loss is returned via aux dict (key 'kl_loss').
    Recon branch has no LN — VAE KL already constrains distribution toward N(0,1).
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.recon_proj_dim = int(kwargs.get("recon_proj_dim", dim))
        self.output_dim = dim + self.recon_proj_dim

        # VAE heads: recon 768 -> recon_proj_dim
        vae_head_adapter = kwargs.get("vae_head_adapter", "LinearAdapter")
        vae_head_adapter_kwargs = dict(kwargs.get("vae_head_adapter_kwargs", {}))
        self.mu_head = build_adapter(
            dim, vae_head_adapter, out_dim=self.recon_proj_dim, **vae_head_adapter_kwargs
        )
        self.logvar_head = build_adapter(
            dim, vae_head_adapter, out_dim=self.recon_proj_dim, **vae_head_adapter_kwargs
        )
        self.logvar_clamp = tuple(kwargs.get("logvar_clamp", (-10.0, 10.0)))

        # KL reduction
        kl_reduce = kwargs.get("kl_reduce", "sum_then_mean")
        if kl_reduce not in ("sum_then_mean", "mean"):
            raise ValueError(f"ConcatMergerVAE: unknown kl_reduce='{kl_reduce}'")
        self.kl_reduce = kl_reduce

        # Sampling mode
        vae_sample_mode = kwargs.get("vae_sample_mode", "auto")
        if vae_sample_mode not in ("auto", "sample", "mode"):
            raise ValueError(f"ConcatMergerVAE: unknown vae_sample_mode='{vae_sample_mode}'")
        self.vae_sample_mode = vae_sample_mode

        # Zero-init VAE heads so initial output ≈ 0 (mu≈0, logvar≈0 → std≈1)
        for head in [self.mu_head, self.logvar_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Semantic LN (recon has no LN — KL constrains distribution)
        self.sem_ln = nn.LayerNorm(dim, elementwise_affine=False)

        logger.info(
            f"ConcatMergerVAE: dim={dim}, recon_proj_dim={self.recon_proj_dim}, "
            f"output_dim={self.output_dim}, vae_head_adapter={vae_head_adapter}, "
            f"kl_reduce={kl_reduce}, vae_sample_mode={vae_sample_mode}, "
            f"logvar_clamp={self.logvar_clamp}"
        )

    def sementic_proj(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Extract semantic component from merged tokens (first dim dimensions)."""
        return merged_tokens[..., :self.output_dim - self.recon_proj_dim]

    def extract_recon(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Extract recon component from merged tokens (last recon_proj_dim dimensions)."""
        return merged_tokens[..., -self.recon_proj_dim:]

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, **kwargs) -> tuple:
        # VAE on recon branch
        mu = self.mu_head(recon_tokens)
        logvar = self.logvar_head(recon_tokens)
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])

        # Determine whether to sample
        force_sample = kwargs.get("force_sample", None)
        if force_sample is not None:
            do_sample = force_sample
        elif self.vae_sample_mode == "sample":
            do_sample = True
        elif self.vae_sample_mode == "mode":
            do_sample = False
        else:  # "auto"
            do_sample = self.training

        if do_sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL loss
        kl_loss = None
        if do_sample:
            kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
            if self.kl_reduce == "mean":
                kl_loss = kl_per_dim.mean()
            else:  # "sum_then_mean"
                kl_loss = kl_per_dim.sum(dim=-1).mean()

        sem = self.sem_ln(sementic_tokens)
        merged = torch.cat([sem, z], dim=-1)

        # Store VAE info in kwargs for caller to retrieve
        # Return format matches ConcatMerger: (merged, recon_out, merged_for_norm)
        # Plus kl_loss as 4th element when available
        if kl_loss is not None:
            return merged, merged, merged, {"kl_loss": kl_loss, "mu": mu, "logvar": logvar}
        return merged, merged, merged


class AddMerger(nn.Module):
    """Merge tokens from two backbones via LN + weighted add.

    merged = LN(sem) + recon_weight * LN(recon)
    sementic_hidden_states = sementic_proj(merged)
    recon_hidden_states = merged

    recon_weight controls variance ratio: sem_var_fraction = 1 / (1 + recon_weight^2).
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.recon_weight = float(kwargs.get("recon_weight", 1.0))

        self.sem_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.recon_ln = nn.LayerNorm(dim, elementwise_affine=False)

        # Post-projection: merged → sementic_hidden_states
        sementic_proj_cls = kwargs.get("sementic_proj_cls", "LinearAdapter")
        sementic_kw = dict(kwargs.get("sementic_proj_kwargs", {}))
        if "dim" not in sementic_kw:
            sementic_kw["dim"] = dim
        self.sementic_proj = build_adapter(sementic_kw.pop("dim", dim), cls_or_name=sementic_proj_cls, **sementic_kw)

        sem_frac = 1.0 / (1.0 + self.recon_weight ** 2)
        logger.info(f"AddMerger: recon_weight={self.recon_weight:.4f}, sem_var_fraction={sem_frac:.4f}")

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("AddMerger: training full model")
            self.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, **kwargs) -> tuple:
        merged = self.sem_ln(sementic_tokens) + self.recon_weight * self.recon_ln(recon_tokens)
        sementic_hidden_states = self.sementic_proj(merged)
        recon_hidden_states = merged

        # Return format compatible with TokenMerger: (sementic, recon, merged)
        return sementic_hidden_states, recon_hidden_states, merged


class DimKMerger(nn.Module):
    """Merge tokens from two backbones. Supports concat+linear, add, learned_weight."""

    def __init__(self, dim: int, k: int =2 ,merger_type: str = "concat", **kwargs):
        super().__init__()
        recon_proj_cls = kwargs.get("recon_proj_cls")
        recon_kw = kwargs.get("recon_proj_kwargs")
        self.recon_proj = build_adapter(dim, cls_or_name=recon_proj_cls, out_dim=k, **recon_kw)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.k = k

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("TokenMerger: training full model")
            self.requires_grad_(True)
        elif mode == "recon_proj_only":
            logger.info("TokenMerger: training recon proj only")
            for p in self.recon_proj.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown training mode: {mode}")


    # def get_last_layer(self) -> torch.nn.Parameter:
    #     return self.recon_proj.get_last_layer()

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, return_distribution: bool = False) -> torch.Tensor:
        b, n ,d = sementic_tokens.shape
        sementic_tokens = self.norm(sementic_tokens)
        recon_tokens = torch.cat((self.recon_proj(recon_tokens), torch.zeros((b, n, d - self.k)).to(recon_tokens)), dim= -1)# [dim:k]
        hidden_states = sementic_tokens+ recon_tokens

        return hidden_states, hidden_states, hidden_states


class NoiseMerger(nn.Module):
    """Merge tokens from two backbones. """

    def __init__(self, dim: int, merger_type: str = "concat", **kwargs):
        super().__init__()
        self.merger_type = merger_type

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("NoiseMerger: training full model")
            self.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def noising(self, x: torch.Tensor, noise_tau: torch.Tensor) -> torch.Tensor:
        noise = noise_tau * torch.randn_like(x)
        return x + noise

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor) -> torch.Tensor:
        sementic_tokens = self.norm(sementic_tokens)
        recon_tokens = self.norm(recon_tokens)
        # if self.training:
        #     logger.debug(f"Noising sementic tokens")
        #     recon_tokens = self.noising(sementic_tokens, recon_tokens)
        # else:
        #     recon_tokens = sementic_tokens
        recon_tokens = self.noising(sementic_tokens, recon_tokens)
        return sementic_tokens, recon_tokens, sementic_tokens

class InterpolationMerger(nn.Module):
    """Merge tokens from two backbones. Supports concat+linear, add, learned_weight."""

    def __init__(self, dim: int, merger_type: str = "concat", **kwargs):
        super().__init__()
        self.merge_linear_interpolation = kwargs.get("merge_linear_interpolation", False)
        self.distribution = kwargs.get("distribution", 'uniform')
        self.distribution_kwargs = kwargs.get("distribution_kwargs", {})
        self.use_add = kwargs.get("use_add", True)

        self.force_semantic = kwargs.get("force_semantic", False)
        logger.debug(f"force semantic:{self.force_semantic}")
    def linear_interpolation_random(self, token1, token2, distribution='uniform', **kwargs):
        """
        对两个 Token 进行基于向量相加的随机线性插值。
        每个 batch 样本使用独立的 t，以增强 Mixup 正则化效果。

        参数:
            token1: 第一个 token (Tensor, 首维为 batch). origin point
            token2: 第二个 token (Tensor, 首维为 batch).
            distribution: 'uniform' (均匀分布) 或 'beta' (Mixup常用)
            alpha: Beta 分布的参数 (仅在 distribution='beta' 时有效)
        """
        batch_size = token1.size(0)
        device = token1.device

        # 1. 按 batch 随机采样 t (每个样本独立)
        if distribution == 'uniform':
            t = torch.rand(batch_size, device=device)
        elif distribution == "bool":
            t = torch.randint(0, 2, (batch_size,), device=device).float()  # float 以支持插值及 mean/std
            logger.debug(f"Bool distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == 'beta':
            alpha = kwargs.get("alpha", 1.0)
            beta = kwargs.get("beta", alpha)
            logger.debug(f"Beta distribution: alpha={alpha}, beta={beta}")
            t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
            logger.debug(f"Beta distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == 'token_beta':
            alpha = kwargs.get("alpha", 1.0)
            beta = kwargs.get("beta", alpha)
            logger.debug(f"Dim beta distribution: alpha={alpha}, beta={beta}")
            b,n,c = token1.shape
            if beta == 1.0:
                t = torch.rand((b,n), device=device, dtype=token1.dtype).pow(1.0 / alpha)
            elif alpha == 1.0:
                t = 1.0 - torch.rand((b,n), device=device, dtype=token1.dtype).pow(1.0 / beta)
            else:
                t = torch.distributions.Beta(alpha, beta).sample((b,n)).to(device)
            logger.debug(f"Beta distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == 'dim_beta':
            alpha = kwargs.get("alpha", 1.0)
            beta = kwargs.get("beta", alpha)
            logger.debug(f"Dim beta distribution: alpha={alpha}, beta={beta}")
            if beta == 1.0:
                t = torch.rand(token1.shape, device=device, dtype=token1.dtype).pow(1.0 / alpha)
            elif alpha == 1.0:
                t = 1.0 - torch.rand(token1.shape, device=device, dtype=token1.dtype).pow(1.0 / beta)
            else:
                t = torch.distributions.Beta(alpha, beta).sample(token1.shape).to(device)
            logger.debug(f"Beta distribution t: {t.mean().item()}, {t.std().item()}")
        elif distribution == 'dim_randn':
            logger.debug(f"Dim randn distribution")
            diff = torch.abs(token1 - token2)
            mixed_token = token2 + diff * torch.randn_like(diff)
            return mixed_token
        else:
            raise ValueError("Unknown distribution")

        # 2. reshape 以广播到 (B, ..., D): t -> (B, ..1)
        t = t.view(*t.shape, *([1] * (token1.dim() - t.dim())))
        logger.debug(f"t shape: {t.shape}")
        # 3. 线性插值: (1-t)*A + t*B
        mixed_token = (1 - t) * token1 + t * token2
        return mixed_token

    def forward(self, sementic_tokens: torch.Tensor, recon_tokens: torch.Tensor, mixer_ratio=None, eval_random=False, **kwargs) -> torch.Tensor:

        start_sementic_hidden_states = sementic_tokens

        if self.force_semantic:
            sementic_hidden_states = start_sementic_hidden_states
        elif self.use_add:
            sementic_hidden_states = recon_tokens + start_sementic_hidden_states
        else:
            logger.debug(f"Add is False, use recon tokens")
            sementic_hidden_states = recon_tokens
        logger.debug(f"sementic_hidden_states norm: {sementic_hidden_states.norm(dim=-1).mean().item()}, start_sementic_hidden_states norm: {start_sementic_hidden_states.norm(dim=-1).mean().item()}")
        if (self.training and self.merge_linear_interpolation) or eval_random:
            logger.debug(f"Linear interpolation")
            hidden_states = self.linear_interpolation_random(start_sementic_hidden_states, sementic_hidden_states, distribution=self.distribution, **self.distribution_kwargs)
        else:
            if mixer_ratio is not None:
                hidden_states = (1 - mixer_ratio) * start_sementic_hidden_states + mixer_ratio * sementic_hidden_states
            else:
                hidden_states = sementic_hidden_states

        return sementic_hidden_states, hidden_states, hidden_states



@register_encoder()
class SigLIP2wTwoBackbone(nn.Module):
    """
    Two SigLIP vision backbones loaded from different weights, with a merger to combine tokens.
    """

    def __init__(
        self,
        sementic_model_name: str = None,
        recon_model_name: str = None,
        num_tokens: int = 256,
        merger_type: str = "concat",
        recon_vae_enabled: bool = False,
        **kwargs,
    ):
        super().__init__()

        if not sementic_model_name or not recon_model_name:
            raise ValueError("SigLIP2wTwoBackbone requires sementic_model_name and recon_model_name (in encoder_params)")
        logger.info(f"SigLIP2wTwoBackbone init: sementic_model_name={sementic_model_name}, recon_model_name={recon_model_name}")
        self.sementic_model_name = sementic_model_name
        self.recon_model_name = recon_model_name
        self.num_tokens = num_tokens
        self.recon_vae_enabled = recon_vae_enabled
        sementic_backbone = SiglipModel.from_pretrained(sementic_model_name).vision_model
        logger.info(f"sementic_model_name={sementic_model_name}")
        logger.info(f"recon_model_name={recon_model_name}")
        self.sementic_backbone = sementic_backbone

        recon_random_init = kwargs.get("recon_random_init", False)
        if recon_random_init:
            recon_backbone = self.random_init_backbone(recon_model_name)
            self.recon_backbone = recon_backbone
            logger.warning(f"recon backbone random initialized")
        else:
            recon_backbone = SiglipModel.from_pretrained(recon_model_name).vision_model
            self.recon_backbone = recon_backbone


        recon_zero_proj = kwargs.get("recon_zero_proj", False)
        if recon_zero_proj:
            self.recon_zero_proj = nn.Linear(self.recon_backbone.config.hidden_size, self.sementic_backbone.config.hidden_size)
            self.recon_zero_proj.weight.data.zero_()
            self.recon_zero_proj.bias.data.zero_()
        else:
            self.recon_zero_proj = None

        hidden_size = sementic_backbone.config.hidden_size # 以语义的hidden size作为hidden size
        # assert recon_backbone.config.hidden_size == hidden_size, "Both backbones must have same hidden_size"

        self.hidden_size = hidden_size
        self.recon_hidden_size = recon_backbone.config.hidden_size

        self.patch_size = sementic_backbone.config.patch_size

        recon_post_norm_type = kwargs.get("recon_post_norm_type", "parameter_free")
        if recon_post_norm_type == "parameter_free":
            self.recon_post_layernorm = nn.LayerNorm(self.recon_hidden_size, elementwise_affine=False)
        elif recon_post_norm_type == "no_norm":
            self.recon_post_layernorm = None
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")

        sementic_post_norm_type = kwargs.get("sementic_post_norm_type", "siglip_post_norm")
        if sementic_post_norm_type == "siglip_post_norm":
            self.sementic_post_layernorm = sementic_backbone.post_layernorm
        elif sementic_post_norm_type == "no_norm":
            self.sementic_post_layernorm = None
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")

        merger_cls = kwargs.get("merger_cls", "TokenMerger")
        if merger_cls == "TokenMerger":
            self.merger = TokenMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "NoiseMerger":
            self.merger = NoiseMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "InterpolationMerger":
            self.merger = InterpolationMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "DimKMerger":
            self.merger = DimKMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "AddMerger":
            self.merger = AddMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "ConcatMerger":
            self.merger = ConcatMerger(hidden_size, **kwargs.get("merger_kwargs", {}))
        elif merger_cls == "ConcatMergerVAE":
            self.merger = ConcatMergerVAE(hidden_size, **kwargs.get("merger_kwargs", {}))
        else:
            raise ValueError(f"Unknown merger_cls: {merger_cls}")

        # Override recon_hidden_size if merger exposes output_dim (e.g. ConcatMerger)
        if hasattr(self.merger, 'output_dim'):
            self.recon_hidden_size = self.merger.output_dim

        self.image_features_use_merged = kwargs.get("image_features_use_merged", False)
        self.return_distribution = kwargs.get("return_distribution", False)

        # Debug mode: replace semantic/recon tokens with precomputed defaults before merger
        self.debug_mode = kwargs.get("debug_mode", None)  # "replace_semantic" | "replace_recon" | None
        if self.debug_mode is not None:
            # Each path points to a normalization_stats.pt with {"mean": [256, 768], "var": ...}
            debug_sementic_stats_path = kwargs.get("debug_sementic_stats_path", None)
            debug_recon_stats_path = kwargs.get("debug_recon_stats_path", None)
            logger.info(f"Debug mode: {self.debug_mode}")
            if debug_sementic_stats_path is not None:
                stats = torch.load(debug_sementic_stats_path, map_location="cpu")
                self.register_buffer("debug_default_sementic_tokens", stats["mean"])  # [256, 768]
                logger.info(f"Loaded default sementic tokens mean from {debug_sementic_stats_path}: {self.debug_default_sementic_tokens.shape}")
            else:
                self.debug_default_sementic_tokens = None
            if debug_recon_stats_path is not None:
                stats = torch.load(debug_recon_stats_path, map_location="cpu")
                self.register_buffer("debug_default_recon_tokens", stats["mean"])  # [256, 768]
                logger.info(f"Loaded default recon tokens mean from {debug_recon_stats_path}: {self.debug_default_recon_tokens.shape}")
            else:
                self.debug_default_recon_tokens = None
        else:
            self.debug_default_sementic_tokens = None
            self.debug_default_recon_tokens = None

        self.use_global_token = kwargs.get("use_global_token", False)

        # 是否计算 MAP head 的 query-grid attention weight 作为 per-token 语义重要度
        # 若启用，forward 的 aux 中会额外返回 "map_attn_weights": (B, N) 的张量
        self.use_map_attn_weight = kwargs.get("use_map_attn_weight", False)
        # 聚合多头的方式: "mean"（默认）或 "max"
        self.map_attn_weight_reduce = kwargs.get("map_attn_weight_reduce", "mean")

        # Setup VAE heads
        if self.recon_vae_enabled:
            vae_head_cls = kwargs.get("vae_head_cls", "LinearAdapter")
            vae_head_kwargs = kwargs.get("vae_head_kwargs", {})
            self.mu_head = build_adapter(self.hidden_size, vae_head_cls, **vae_head_kwargs)
            self.logvar_head = build_adapter(self.hidden_size, vae_head_cls, **vae_head_kwargs)
            self.logvar_clamp = kwargs.get("vae_head_kwargs", (-10, 10))
            logger.info(f"SigLIP2wVAE: mu_head and logvar_head using {vae_head_cls}")
            vae_head_pretrained_path = kwargs.get("vae_head_pretrained_path", None)
            if vae_head_pretrained_path is not None:
                ckpt = torch.load(vae_head_pretrained_path, map_location="cpu")

                # 支持 PatchReparam 完整 checkpoint 或直接的 vae head checkpoint
                if "mu_head" in ckpt and "logvar_head" in ckpt:
                    # 直接的 vae head checkpoint
                    self.mu_head.load_state_dict(ckpt["mu_head"])
                    self.logvar_head.load_state_dict(ckpt["logvar_head"])
                elif "ema" in ckpt:
                    mu_sd = {k.split("mu_head.")[-1]: v for k, v in ckpt["ema"].items() if "mu_head" in k}
                    logvar_sd = {k.split("logvar_head.")[-1]: v for k, v in ckpt["ema"].items() if "logvar_head" in k}
                    if mu_sd:
                        self.mu_head.load_state_dict(mu_sd)
                        logger.info("Load mu head from ema")
                    if logvar_sd:
                        self.logvar_head.load_state_dict(logvar_sd)
                        logger.info("Load var head from ema")
                else:
                    raise ValueError
                logger.info(f"Loaded VAE heads from {vae_head_pretrained_path}")
            # Apply initialization based on vae_head_init
            # self._init_vae_heads(self.vae_head_init)

    def random_init_backbone(self, model_name:str, **kwargs):

        config = SiglipConfig.from_pretrained(model_name)
        full_model = SiglipVisionModel(config.vision_config)
        model = full_model.vision_model

        # zero_output = kwargs.get("zero_output", False)

        return model

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("SigLIP2wTwoBackbone: training full model")
            self.requires_grad_(True)
        elif mode == "merger_only":
            logger.info("SigLIP2wTwoBackbone: training merger only")
            for p in self.merger.parameters():
                p.requires_grad = True
        elif mode == "recon_only":
            self.training_mode = "recon_only"
            logger.info("SigLIP2wTwoBackbone: training recon only")
            for p in self.recon_backbone.parameters():
                p.requires_grad = True
            logger.warning("SigLIP2wTwoBackbone: training recon backbone only, siglip head and postnorm will be frozen")
            for param in self.recon_backbone.head.parameters():
                param.requires_grad = False
                logger.debug(f"siglip head {param.shape} requires_grad: {param.requires_grad}")
            for param in self.recon_backbone.post_layernorm.parameters():
                param.requires_grad = False
                logger.debug(f"siglip postnorm {param.shape} requires_grad: {param.requires_grad}")
            if self.recon_zero_proj is not None:
                self.recon_zero_proj.requires_grad_(True)
                logger.debug(f"recon zero proj requires_grad")
        elif mode == "merger_recon":
            logger.info("SigLIP2wTwoBackbone: training merger and recon backbone only")
            for p in self.merger.parameters():
                p.requires_grad = True
            for p in self.recon_backbone.parameters():
                p.requires_grad = True
            logger.warning("SigLIP2wTwoBackbone: training recon backbone only, siglip head and postnorm will be frozen")
            for param in self.recon_backbone.head.parameters():
                param.requires_grad = False
                logger.debug(f"siglip head {param.shape} requires_grad: {param.requires_grad}")
            for param in self.recon_backbone.post_layernorm.parameters():
                param.requires_grad = False
                logger.debug(f"siglip postnorm {param.shape} requires_grad: {param.requires_grad}")
        elif mode == "merger_recon_patchemb":
            logger.info("SigLIP2wTwoBackbone: training merger and recon backbone patch_embed only")
            for p in self.merger.parameters():
                p.requires_grad = True
            if hasattr(self.recon_backbone, "embeddings") and self.recon_backbone.embeddings is not None:
                for p in self.recon_backbone.embeddings.parameters():
                    p.requires_grad = True
        elif mode == "recon_proj_only":
            logger.info("SigLIP2wTwoBackbone: training recon head only")
            self.merger.set_training_mode(mode="recon_proj_only")
        elif mode == "patchemb":
            logger.info("SigLIP2wTwoBackbone: training patch embedding")
            for backbone in [self.sementic_backbone, self.recon_backbone]:
                if hasattr(backbone, "embeddings") and backbone.embeddings is not None:
                    for p in backbone.embeddings.parameters():
                        p.requires_grad = True
        elif mode == "frozen":
            self.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        if return_stats:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable_params": trainable, "total_params": total}
        return None

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        _, aux = self.forward(images, return_aux=True)
        sementic_hidden_states = aux["sementic_tokens"]
        return self.sementic_backbone.head(sementic_hidden_states)

    # def output_head_forward(self, hidden_states: torch.Tensor, mean_first: bool = True) -> torch.Tensor:
    #     if not self.need_output_head or not hasattr(self.model, "head") or self.model.head is None:
    #         if hidden_states.dim() == 3:
    #             return hidden_states.mean(dim=1)
    #         return hidden_states
    #     if hidden_states.dim() == 4:
    #         hidden_states = hidden_states.flatten(2).transpose(1, 2)
    #     if self.output_head_type == "siglip_MAP":
    #         return self.model.head(hidden_states)
    #     elif self.output_head_type == "siglip_MAP_with_postnorm":
    #         return self.model.head(self.model.post_layernorm(hidden_states))
    #     raise ValueError(f"Unknown output_head_type: {self.output_head_type}")

    def get_last_layer(self) -> torch.nn.Parameter:
        if hasattr(self.merger, "get_last_layer"):
            return self.merger.get_last_layer()

        elif hasattr(self, "training_mode") and self.training_mode == "recon_only":
            return self.recon_backbone.encoder.layers[-1].mlp.fc2.weight
        else:
            logger.warning("No last layer found for merger")
            return None

    def get_recon_last_layer(self) -> torch.nn.Parameter:
        """Return the last trainable parameter in the recon token pathway."""
        # Try merger's recon-specific last layer first
        if hasattr(self.merger, "get_recon_last_layer"):
            return self.merger.get_recon_last_layer()
        # Fallback: merger's general last layer
        if hasattr(self.merger, "get_last_layer"):
            return self.merger.get_last_layer()
        # Fallback: recon backbone's last layer
        if hasattr(self, "recon_backbone"):
            return self.recon_backbone.encoder.layers[-1].mlp.fc2.weight
        return None

    def _get_tokens(self, backbone, images: torch.Tensor):
        try:
            hidden_states = backbone.embeddings(images, interpolate_pos_encoding=True)
        except TypeError:
            # 为了兼容llava
            hidden_states = backbone.embeddings(images)
        encoder_out = backbone.encoder(inputs_embeds=hidden_states, output_hidden_states=True)
        pre_postnorm = encoder_out.last_hidden_state
        return pre_postnorm

    def get_recon_tokens(self, merged_tokens):
        # print("debug mode:", self.debug_mode)
        # print("debug default sementic tokens:", self.debug_default_sementic_tokens is not None)
        if self.debug_mode == "replace_semantic" and self.debug_default_sementic_tokens is not None:
            print("debug mode: replace semantic")

            sementic_tokens = merged_tokens[:,:,:768]
            recon_tokens = merged_tokens[:,:,768:]
            default_sem = self.debug_default_sementic_tokens
            # Handle [C, H, W] -> [N, D] conversion if needed
            if default_sem.dim() == 3:
                C, H, W = default_sem.shape
                default_sem = default_sem.permute(1, 2, 0).reshape(H * W, C)  # [H*W, C]
            sementic_tokens = default_sem.unsqueeze(0).expand_as(sementic_tokens).to(sementic_tokens.dtype)
            recon_hidden_states = torch.cat([sementic_tokens, recon_tokens], dim=2)
            return recon_hidden_states
        elif self.debug_mode == "zero_sementic":
            print("debug mode: zero semantic")
            sementic_tokens = merged_tokens[:,:,:768]
            sementic_tokens = torch.zeros_like(sementic_tokens)
            recon_tokens = merged_tokens[:,:,768:]
            recon_hidden_states = torch.cat([sementic_tokens, recon_tokens], dim=2)
            return recon_hidden_states
        if hasattr(self.merger, "extract_recon"):
            recon_hidden_states = self.merger.extract_recon(merged_tokens)
        elif hasattr(self.merger, "recon_proj") and self.merger.recon_proj is not None:
            recon_hidden_states = self.merger.recon_proj(merged_tokens)
        else:
            recon_hidden_states = merged_tokens

        if self.recon_post_layernorm:
            return self.recon_post_layernorm(recon_hidden_states)
        else:
            return recon_hidden_states

    def get_sementic_tokens_for_recon(self, merged_tokens):
        sementic_hidden_states = self.merger.sementic_proj(merged_tokens)
        if self.sementic_post_layernorm:
            return self.sementic_post_layernorm(sementic_hidden_states)
        else:
            return sementic_hidden_states
    def get_recon_postnorm(self, tokens):
        return self.recon_post_layernorm(tokens)

    @torch.no_grad()
    def get_map_query_grid_similarity(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        """计算 sementic backbone MAP head 的 query（probe）与 grid token 的相似度（attention softmax 权重）。

        SiglipMultiheadAttentionPoolingHead 使用一个可学习的 probe 作为 query，
        对所有 grid token（key/value）做 multi-head attention，其 softmax 权重即
        反映了 query 对每个 grid token 的关注程度。

        Args:
            grid_tokens: (B, N, D) — 经过 sementic_post_layernorm 之后的 grid token 序列
                         （即 post-postnorm），与 SigLIP MAP head 实际接收的输入保持一致。

        Returns:
            attn_weights: (B, num_heads, N) — 每个 attention head 对应的 softmax 权重，
                          即 MAP query（probe）与每个 grid token 的相似度分布。
        """
        head = self.sementic_backbone.head  # SiglipMultiheadAttentionPoolingHead
        assert hasattr(head, "probe") and hasattr(head, "attention"), (
            "sementic_backbone.head 不是 SiglipMultiheadAttentionPoolingHead，缺少 probe/attention 属性"
        )

        B = grid_tokens.shape[0]
        probe = head.probe.repeat(B, 1, 1)  # (B, 1, D)

        # need_weights=True 返回 softmax 后的 attention 权重
        # average_attn_weights=False 保留每个 head 的独立权重
        _, attn_weights = head.attention(
            probe,        # query: (B, 1, D)
            grid_tokens,  # key:   (B, N, D)
            grid_tokens,  # value: (B, N, D)
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_weights shape: (B, num_heads, 1, N) -> squeeze query dim -> (B, num_heads, N)
        attn_weights = attn_weights.squeeze(2)
        return attn_weights
    def _vae_sample(self, hidden_states: torch.Tensor):
        """Perform VAE sampling: pre_norm -> mu/logvar -> sampling."""
        # pre_norm before VAE head


        # Compute mu and logvar
        mu = self.mu_head(hidden_states)
        logvar = self.logvar_head(hidden_states)
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])

        # Reparameterization trick (training) or deterministic (eval)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar
    def forward(self, images: torch.Tensor, image_features_mode: str = None, **kwargs):
        """
        image_features_mode: override which tokens to use as image_features for reconstruction.
            - "recon": recon_hidden_states (post layernorm)
            - "merged": merged_hidden_states
            - "sementic": sementic_hidden_states (post layernorm)
            - None: use self.image_features_use_merged (training default)
        """
        # print("images dtype:", images.dtype)
        sementic_tokens = self._get_tokens(self.sementic_backbone,  images)
        recon_tokens = self._get_tokens(self.recon_backbone,  images)

        if self.recon_vae_enabled:
            recon_tokens, _, _ = self._vae_sample(recon_tokens)

        if self.recon_zero_proj is not None:
            # zero out the recon tokens
            recon_tokens = self.recon_zero_proj(recon_tokens)
        # print("recon_tokens dtype:", recon_tokens.dtype)

        # Debug mode: replace tokens with precomputed defaults before merger
        if self.debug_mode == "zero_recon":
            recon_tokens = torch.zeros_like(recon_tokens)
        elif self.debug_mode == "zero_sementic":
            sementic_tokens = torch.zeros_like(sementic_tokens)
        elif self.debug_mode == "replace_recon" and self.debug_default_recon_tokens is not None:
            default_recon = self.debug_default_recon_tokens
            if default_recon.dim() == 3:
                C, H, W = default_recon.shape
                default_recon = default_recon.permute(1, 2, 0).reshape(H * W, C)
            recon_tokens = default_recon.unsqueeze(0).expand_as(recon_tokens).to(recon_tokens.dtype)
        elif self.debug_mode == "replace_semantic" and self.debug_default_sementic_tokens is not None:
            default_sem = self.debug_default_sementic_tokens
            if default_sem.dim() == 3:
                C, H, W = default_sem.shape
                default_sem = default_sem.permute(1, 2, 0).reshape(H * W, C)
            sementic_tokens = default_sem.unsqueeze(0).expand_as(sementic_tokens).to(sementic_tokens.dtype)

        merger_output = self.merger(sementic_tokens, recon_tokens, return_distribution=self.return_distribution)
        merger_vae_info = None
        if len(merger_output) == 4 and isinstance(merger_output[3], dict):
            sementic_hidden_states, recon_hidden_states, merged_hidden_states, merger_vae_info = merger_output
        elif self.return_distribution:
            sementic_hidden_states, recon_hidden_states, merged_hidden_states, t = merger_output
        else:
            sementic_hidden_states, recon_hidden_states, merged_hidden_states = merger_output
        # print("sementic_hidden_states dtype:", sementic_hidden_states.dtype)
        sementic_post = self.sementic_post_layernorm(sementic_hidden_states) if self.sementic_post_layernorm is not None else sementic_hidden_states
        recon_post = self.recon_post_layernorm(recon_hidden_states) if self.recon_post_layernorm is not None else recon_hidden_states
        # origin_recon_post = self.recon_post_layernorm(sementic_tokens)
        # new_recon_post = origin_recon_post + 0.8 * torch.randn_like(origin_recon_post)
        # new_recon_post = self.recon_post_layernorm(merged_hidden_states)
        # diff_full = new_recon_post - origin_recon_post


        # noise = 0.8 * torch.randn_like(new_recon_post)
        # merged_hidden_states = origin_recon_post
        # merged_hidden_states += noise
        # print((merged_hidden_states - origin_recon_post)[1].abs().mean().item())
        # noise 与 abs(diff_full) 相照应：diff 越大的维度，加的 noise 幅度越大
        # weighted noise
        # abs_diff = diff_full.abs()
        # weight = abs_diff / (abs_diff.mean() + 1e-8)  # 归一化使整体尺度与原先 0.8 相当
        # print(weight.max().item(), weight.min().item())
        # merged_hidden_states += 0.8 * weight * torch.randn_like(new_recon_post, device=new_recon_post.device, dtype=new_recon_post.dtype)

        # 每个 token 中 diff 最大的维度置为与 origin_recon_post 一样（使 diff=0），其余保持不变
        # abs_diff = diff_full.abs()
        # max_dim_per_token = abs_diff.argmax(dim=-1)  # (B, N)
        # batch_idx = torch.arange(abs_diff.shape[0], device=merged_hidden_states.device).unsqueeze(1).expand(-1, abs_diff.shape[1])
        # token_idx = torch.arange(abs_diff.shape[1], device=merged_hidden_states.device).unsqueeze(0).expand(abs_diff.shape[0], -1)
        # new_recon_post[batch_idx, token_idx, max_dim_per_token] = origin_recon_post[batch_idx, token_idx, max_dim_per_token]
        # merged_hidden_states[batch_idx, token_idx, max_dim_per_token] = origin_recon_post[batch_idx, token_idx, max_dim_per_token]

        # merged_hidden_states = origin_recon_post
        # print("noise", (merged_hidden_states - origin_recon_post)[1].abs().mean().item())
        # merged_hidden_states = self.recon_post_layernorm(merged_hidden_states)
        # print("postnorm", (merged_hidden_states - origin_recon_post)[1].abs().mean().item())
        # logger.debug("noise")
        # noise= 0.8 * torch.randn_like(new_recon_post)
        # logger.debug(f"noise max {noise.abs().max().item()}")
        # logger.debug(f"noise mean {noise.abs().mean().item()}")
        # logger.debug("dim 0")
        # logger.debug((new_recon_post - origin_recon_post)[:,:,0])
        # logger.debug((new_recon_post - origin_recon_post)[:,:,0].abs().max().item())
        # logger.debug("dim 7")
        # logger.debug((new_recon_post - origin_recon_post)[:,:,7])
        # logger.debug((new_recon_post - origin_recon_post)[:,:,7].abs().max().item())
        # for tok_idx in [0, 1, 120]:
        #     diff = (new_recon_post - origin_recon_post)[:, tok_idx]  # (B, D)
        #     abs_diff = diff.abs()
        #     max_val = abs_diff.max().item()
        #     max_flat = abs_diff.argmax().item()
        #     batch_idx, dim_idx = divmod(max_flat, diff.shape[-1])
        #     logger.debug(f"token {tok_idx}: max={max_val:.6f} at dim={dim_idx} (batch={batch_idx})")
        #     # 每个样本该 token 的最大值出现在哪个维度
        #     max_dim_per_sample = abs_diff.argmax(dim=-1).cpu().tolist()
        #     logger.debug(f"token {tok_idx} per-sample max dim: {max_dim_per_sample}")
        # logger.debug("batch 0")
        # logger.debug((new_recon_post - origin_recon_post)[0])
        # logger.debug((new_recon_post - origin_recon_post)[0].abs().max().item())
        # logger.debug((new_recon_post - origin_recon_post)[0].abs().mean().item())
        # logger.debug("batch 1")
        # logger.debug((new_recon_post - origin_recon_post)[1])
        # logger.debug((new_recon_post - origin_recon_post)[1].abs().max().item())
        # logger.debug((new_recon_post - origin_recon_post)[1].abs().mean().item())

        # # 每个样本：最大diff值出现在哪个token；维度平均diff最大的token
        # abs_diff = diff_full.abs()
        # B, N, D = abs_diff.shape
        # for b in range(B):
        #     # 最大 diff 出现在哪个 token（对样本 b 在 (token,dim) 上找全局最大）
        #     flat = abs_diff[b].reshape(-1)
        #     max_flat_idx = flat.argmax().item()
        #     tok_max, dim_max = divmod(max_flat_idx, D)
        #     max_val = flat[max_flat_idx].item()
        #     logger.debug(f"sample{b}: max |diff|={max_val:.6f} at token={tok_max} dim={dim_max}")
        #     # 维度平均 diff 最大的 token
        #     mean_per_token = abs_diff[b].mean(dim=-1)  # (N,)
        #     tok_mean_max = mean_per_token.argmax().item()
        #     mean_max_val = mean_per_token[tok_mean_max].item()
        #     logger.debug(f"sample{b}: max mean |diff| over dims={mean_max_val:.6f} at token={tok_mean_max}")

        # # 详细打印 + wandb 可视化（逐样本逐维度逐 token 的 diff）
        # _log_recon_diff_visualization(
        #     diff_full,
        #     step=kwargs.get("global_step"),
        #     token_indices=[0, 1, 50, 100, 120, 200, -1],
        # )
        # # 逐样本、逐 token 的 new_recon_post vs origin_recon_post cosine 相似度 log + 可视化
        # _log_recon_cosine_similarity(
        #     new_recon_post,
        #     origin_recon_post,
        #     step=kwargs.get("global_step"),
        #     token_indices=[0, 1, 50, 100, 120, 200, -1],
        # )

        if image_features_mode == "recon_post":
            image_features = recon_post
        elif image_features_mode == "original_sementic_post":
            image_features = self.recon_post_layernorm(sementic_tokens)
        elif image_features_mode == "mixer0.5":
            _,recon_tmp,_ = self.merger(sementic_tokens, recon_tokens, mixer_ratio=0.5)
            image_features = self.recon_post_layernorm(recon_tmp)
        elif image_features_mode == "random":
            _,recon_tmp,_ = self.merger(sementic_tokens, recon_tokens, eval_random=True)
            image_features = self.recon_post_layernorm(recon_tmp)
        elif self.image_features_use_merged:
            image_features = merged_hidden_states
        else:
            image_features = recon_post

        return_aux = kwargs.get("return_aux", False)

        if self.use_global_token:
            global_token = self.sementic_backbone.head(sementic_hidden_states).unsqueeze(1) #[B,1,D]
            merged_hidden_states_with_global = torch.cat([merged_hidden_states, global_token], dim=1)

        # 计算 MAP head 的 per-token 语义重要度权重 (B, N)
        # 只在 use_map_attn_weight=True 且需要 aux 时计算，避免无谓开销
        map_attn_weights = None
        if self.use_map_attn_weight and (return_aux or not kwargs.get("skip_map_attn", False)):
            # sementic_post: post-postnorm grid tokens, shape (B, N, D)
            # 与 SigLIP MAP head 的实际输入一致，必须用 postnorm 后的 tokens
            # get_map_query_grid_similarity 内部已 @torch.no_grad()
            per_head_weights = self.get_map_query_grid_similarity(sementic_post)  # (B, num_heads, N)
            if self.map_attn_weight_reduce == "max":
                map_attn_weights = per_head_weights.max(dim=1).values  # (B, N)
            else:  # "mean"
                map_attn_weights = per_head_weights.mean(dim=1)  # (B, N)
            logger.debug(f"map_attn_weights shape: {map_attn_weights.shape}, "
                         f"mean={map_attn_weights.mean().item():.6f}, "
                         f"max={map_attn_weights.max().item():.6f}, "
                         f"min={map_attn_weights.min().item():.6f}")

        if return_aux:
            aux = {
                "sementic_tokens": sementic_post,
                "recon_tokens": recon_post,
                "sementic_tokens_raw": sementic_tokens,
                "recon_tokens_raw": recon_tokens,
                "merged_tokens": merged_hidden_states,
                "merged_tokens_with_global": merged_hidden_states_with_global if self.use_global_token else None,
                "sementic_tokens_before_postnorm": sementic_hidden_states,
                "cls_label": t if self.return_distribution else None,
                "debug": torch.cat([sementic_hidden_states, recon_hidden_states], dim=-1),
                "map_attn_weights": map_attn_weights,  # (B, N) or None
                # "sementic_tokens_before_postnorm": sementic_tokens,
            }
            # Propagate VAE info from merger (e.g. ConcatMergerVAE)
            if merger_vae_info is not None:
                aux["kl_loss"] = merger_vae_info.get("kl_loss")
                aux["mu"] = merger_vae_info.get("mu")
                aux["logvar"] = merger_vae_info.get("logvar")
            # logger.debug(f"image_features mean: {image_features.mean().item():.6f}, std: {image_features.std().item():.6f}")
            # logger.debug(f"recon_tokens mean: {self.recon_post_layernorm(recon_hidden_states).mean().item():.6f}, std: {self.recon_post_layernorm(recon_hidden_states).std().item():.6f}")
            return image_features, aux
        return image_features

@register_encoder()
class SigLIP2wFrozenTeacher(nn.Module):
    def __init__(self, model_name:str, teacher_model_name:str, num_tokens=256, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        zero_init = kwargs.get("zero_init", False)
        if zero_init:
            config = SiglipConfig.from_pretrained(self.model_name)
            full_model = SiglipVisionModel(config.vision_config)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)  # 不注册为子模块，保持 state_dict 兼容旧 checkpoint
            logger.warning(f"using zero_init model")
        else:
            full_model = SiglipVisionModel.from_pretrained(self.model_name)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)  # 不注册为子模块，保持 state_dict 兼容旧 checkpoint

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        self.teacher = SiglipModel.from_pretrained(teacher_model_name).vision_model
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        merger_cls = kwargs.get("merger_cls", "TokenMerger")

        if merger_cls == "InterpolationMerger":
            self.merger = InterpolationMerger(self.hidden_size, **kwargs.get("merger_kwargs", {}))
        else:
            raise ValueError(f"Unknown merger_cls: {merger_cls}")
        recon_post_norm_type = kwargs.get("recon_post_norm_type", "parameter_free")
        if recon_post_norm_type == "parameter_free":
            self.recon_post_layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        elif recon_post_norm_type == "no_norm":
            self.recon_post_layernorm = None
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")

        sementic_post_norm_type = kwargs.get("sementic_post_norm_type", "siglip_post_norm")
        if sementic_post_norm_type == "siglip_post_norm":
            self.sementic_post_layernorm = self.model.post_layernorm
        else:
            raise ValueError(f"Unknown post_norm_type: {post_norm_type}")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        full_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        filtered_dict = {k: v for k, v in full_dict.items() if f'{prefix}teacher' not in k}
        if destination is not None:
            destination.clear()
            destination.update(filtered_dict)
            return destination
        return filtered_dict

    def load_state_dict(self, state_dict, strict=True):
        logger.warning("Loading state_dict with strict=False to skip frozen teacher weights.")
        return super().load_state_dict(state_dict, strict=False)

    def set_gradient_checkpointing(self):
        """Enable gradient checkpointing on the SigLIP vision model to save memory.
        使用 SiglipVisionModel (PreTrainedModel) 的标准接口 gradient_checkpointing_enable()。
        """
        enabled = False
        full_model = getattr(self, "_full_vision_model", None)
        if full_model is not None and hasattr(full_model, "gradient_checkpointing_enable"):
            full_model.gradient_checkpointing_enable()
            logger.info("SigLIP2wNorm: gradient checkpointing enabled (via SiglipVisionModel)")
            enabled = True
        # Fallback: 手动设置 gradient_checkpointing 和 _gradient_checkpointing_func
        if not enabled:
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            checkpoint_fn = partial(checkpoint, use_reentrant=False)
            def _enable(module):
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    if getattr(module, "_gradient_checkpointing_func", None) is None:
                        module._gradient_checkpointing_func = checkpoint_fn
            self.model.apply(_enable)
            logger.info("SigLIP2wNorm: gradient checkpointing enabled (fallback)")
            enabled = True
        if not enabled:
            logger.warning("SigLIP2wNorm: gradient checkpointing not available")
    def set_training_mode(self, mode: str = 'full', return_stats: bool = False):
        """
        设置 encoder 的训练模式，控制哪些参数可训练。
        """
        self.requires_grad_(False)
        if mode == 'full':
            logger.info("training full model")
            self.requires_grad_(True)
        elif mode == "no_sigliphead":
            logger.info("siglip head no grad, siglip postnorm no grad")
            self.requires_grad_(True)
            for param in self.model.head.parameters():
                param.requires_grad = False
                logger.debug(f"siglip head {param.shape} requires_grad: {param.requires_grad}")
            for param in self.model.post_layernorm.parameters():
                param.requires_grad = False
                logger.debug(f"siglip postnorm {param.shape} requires_grad: {param.requires_grad}")
        elif mode == "patchemb":
            logger.info("training patch embedding")
            if hasattr(self.model, 'embeddings') and self.model.embeddings is not None:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = True
                    logger.debug(f"patch embedding {param.shape} requires_grad: {param.requires_grad}")
        elif mode == 'frozen':
            self.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        self.teacher.requires_grad_(False)

        if return_stats:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            # 统计self.model之外的参数
            model_params = sum(p.numel() for p in self.model.parameters())
            other_params = trainable_params - model_params


            return {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'model_params': model_params,
                'other_params': other_params,
            }
        return None
    @torch.no_grad()
    def get_image_features(self, images):
        return self.model(images).pooler_output


    def get_last_layer(self):
        return self.model.encoder.layers[-1].mlp.fc2.weight

    def _get_tokens(self, backbone, images: torch.Tensor):
        try:
            hidden_states = backbone.embeddings(images, interpolate_pos_encoding=True)
        except TypeError:
            # 为了兼容llava
            hidden_states = backbone.embeddings(images)
        encoder_out = backbone.encoder(inputs_embeds=hidden_states, output_hidden_states=True)
        pre_postnorm = encoder_out.last_hidden_state
        return pre_postnorm

    def forward(self, images: torch.Tensor, image_features_mode: str = None, **kwargs):
        """
        image_features_mode: override which tokens to use as image_features for reconstruction.
            - "recon": recon_hidden_states (post layernorm)
            - "merged": merged_hidden_states
            - "sementic": sementic_hidden_states (post layernorm)
            - None: use self.image_features_use_merged (training default)
        """
        sementic_tokens = self._get_tokens(self.teacher,  images)
        hidden_states = self._get_tokens(self.model,  images)

        _, _, sampled_hidden_states = self.merger(sementic_tokens, hidden_states)

        sementic_post = self.sementic_post_layernorm(hidden_states)
        recon_post = self.recon_post_layernorm(sampled_hidden_states) if self.recon_post_layernorm is not None else sampled_hidden_states


        image_features = recon_post

        return_aux = kwargs.get("return_aux", False)

        if return_aux:
            aux = {
                "sementic_tokens": sementic_post, #直出的，norm后
                "recon_tokens": recon_post, # 采样的
                "merged_tokens": hidden_states, #直出的norm前
                # "sementic_tokens_before_postnorm": sementic_hidden_states,
                # "debug": torch.cat([sementic_hidden_states, recon_hidden_states], dim=-1),
                # "sementic_tokens_before_postnorm": sementic_tokens,
            }
            # logger.debug(f"image_features mean: {image_features.mean().item():.6f}, std: {image_features.std().item():.6f}")
            # logger.debug(f"recon_tokens mean: {self.recon_post_layernorm(recon_hidden_states).mean().item():.6f}, std: {self.recon_post_layernorm(recon_hidden_states).std().item():.6f}")
            return image_features, aux
        return image_features


@register_encoder()
class SigLIP2wVAE(nn.Module):
    """SigLIP2 encoder with VAE sampling head.

    Architecture:
        encoder_output -> [pre_norm] -> VAE_head(mu, logvar) -> sampling -> [post_norm] -> decoder

    - pre_norm: LayerNorm before VAE head (optional)
    - post_norm: LayerNorm after sampling, before decoder (optional, can be parameter-free or with params)

    VAE head position is determined by which features VAE operates on:
    - vae_enabled=True: VAE operates on encoder output, then post_norm
    - vae_enabled=False: post_norm directly on encoder output (fallback to standard behavior)

    VAE sampling:
    - Training: reparameterization sampling (mu + std * eps)
    - Eval: deterministic output (use mu only)

    KL Loss is computed only in training mode and passed through aux dict.
    """

    def __init__(
        self,
        model_name: str,
        num_tokens: int = 256,
        vae_enabled: bool = True,
        vae_head_adapter: str = "LinearAdapter",
        vae_head_adapter_kwargs: dict = None,
        pre_norm: bool = True,  # LayerNorm before VAE head
        post_norm_type: str = "parameter_free",  # LayerNorm after sampling: "parameter_free", "model_post_norm", "low_rank", "none"
        logvar_clamp: tuple = (-10.0, 10.0),
        vae_head_init: str = "default",  # "default", "zero", "identity"
        kl_reduce: str = "sum_then_mean",  # "sum_then_mean": sum over dim then mean; "mean": global mean over all elements
        vae_hidden_size = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.vae_enabled = vae_enabled
        self.logvar_clamp = logvar_clamp
        self.vae_head_init = vae_head_init
        if kl_reduce not in ("sum_then_mean", "mean"):
            raise ValueError(f"SigLIP2wVAE: unknown kl_reduce='{kl_reduce}', choose 'sum_then_mean' or 'mean'")
        self.kl_reduce = kl_reduce
        logger.info(f"SigLIP2wVAE: kl_reduce='{kl_reduce}'")

        # Load base SigLIP model
        zero_init = kwargs.get("zero_init", False)
        if zero_init:
            config = SiglipConfig.from_pretrained(self.model_name)
            full_model = SiglipVisionModel(config.vision_config)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)
            logger.warning("SigLIP2wVAE: using zero_init model")
        else:
            full_model = SiglipVisionModel.from_pretrained(self.model_name)
            self.model = full_model.vision_model
            object.__setattr__(self, "_full_vision_model", full_model)
        if vae_hidden_size:
            self.hidden_size = vae_hidden_size
            backbone_hidden_size = self.model.config.hidden_size
        else:
            self.hidden_size = self.model.config.hidden_size
            backbone_hidden_size = self.model.config.hidden_size

        self.patch_size = self.model.config.patch_size

        # pre_norm: before VAE head (always parameter-free LayerNorm)
        self.pre_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False) if pre_norm else None
        logger.info(f"SigLIP2wVAE: pre_norm={'enabled' if self.pre_norm else 'disabled'}")

        # post_norm: after sampling, before decoder
        self.post_norm_type = post_norm_type
        if post_norm_type == "none":
            self.post_norm = None
        elif post_norm_type == "model_post_norm":
            self.post_norm = self.model.post_layernorm
        elif post_norm_type == "parameter_free":
            self.post_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        else:
            self.post_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        logger.info(f"SigLIP2wVAE: post_norm type={post_norm_type}")

        # Setup VAE heads
        if self.vae_enabled:
            adapter_kwargs = vae_head_adapter_kwargs or {}
            self.mu_head = build_adapter(backbone_hidden_size, vae_head_adapter, out_dim=self.hidden_size, **adapter_kwargs)
            self.logvar_head = build_adapter(backbone_hidden_size, vae_head_adapter, out_dim=self.hidden_size, **adapter_kwargs)
            logger.info(f"SigLIP2wVAE: mu_head and logvar_head using {vae_head_adapter}")

            # Apply initialization based on vae_head_init
            self._init_vae_heads(self.vae_head_init)
        else:
            self.mu_head = None
            self.logvar_head = None
            logger.info("SigLIP2wVAE: VAE disabled")

        if kwargs.get("gradient_checkpointing", False):
            self.set_gradient_checkpointing()

    def _init_vae_heads(self, init_type: str):
        """
        Initialize VAE heads (mu_head and logvar_head).

        Args:
            init_type: Initialization strategy
                - "default": No special initialization (PyTorch default)
                - "zero": Initialize weights and biases to zero
                    Effect: mu=0, logvar=0 -> z ~ N(0,1) noise (training) or z=0 (eval)
                    Suitable for training VAE from scratch
                - "identity": Initialize mu_head to identity mapping, logvar_head to zero
                    Effect: mu=input, logvar=0 -> z = input + noise (training) or z=input (eval)
                    Suitable for adding VAE on pretrained encoder
        """
        if init_type == "default":
            logger.info("SigLIP2wVAE: VAE heads using default PyTorch initialization")
            return

        def _get_linear_layer(module):
            """Get the linear layer from an adapter module."""
            if hasattr(module, 'linear'):
                return module.linear
            elif hasattr(module, 'mlp'):
                # MLPAdapter: get the last linear layer
                last_idx = getattr(module, '_last_linear_idx', -1)
                if last_idx >= 0:
                    return module.mlp[last_idx]
            elif hasattr(module, 'up'):
                # ResidualAdapter or LowRankAdapter
                return module.up
            return None

        mu_linear = _get_linear_layer(self.mu_head)
        logvar_linear = _get_linear_layer(self.logvar_head)

        if init_type == "zero":
            # Zero initialization: mu=0, logvar=0
            if mu_linear is not None:
                nn.init.zeros_(mu_linear.weight)
                if mu_linear.bias is not None:
                    nn.init.zeros_(mu_linear.bias)
            if logvar_linear is not None:
                nn.init.zeros_(logvar_linear.weight)
                if logvar_linear.bias is not None:
                    nn.init.zeros_(logvar_linear.bias)
            logger.info("SigLIP2wVAE: VAE heads initialized to zero (mu=0, logvar=0 -> z~N(0,1))")

        elif init_type == "identity":
            # Identity initialization for mu_head, zero for logvar_head
            # Effect: z = input + eps (training), z = input (eval)
            if mu_linear is not None:
                if mu_linear.weight.shape[0] == mu_linear.weight.shape[1]:
                    # Square matrix: use identity
                    nn.init.eye_(mu_linear.weight)
                else:
                    # Non-square: use small random values
                    nn.init.normal_(mu_linear.weight, mean=0, std=0.02)
                if mu_linear.bias is not None:
                    nn.init.zeros_(mu_linear.bias)
            if logvar_linear is not None:
                nn.init.zeros_(logvar_linear.weight)
                if logvar_linear.bias is not None:
                    nn.init.zeros_(logvar_linear.bias)
            logger.info("SigLIP2wVAE: VAE heads initialized as identity (mu=input, logvar=0 -> z=input+noise)")
        else:
            logger.warning(f"SigLIP2wVAE: Unknown vae_head_init='{init_type}', using default initialization")

    def set_gradient_checkpointing(self):
        """Enable gradient checkpointing on the SigLIP vision model."""
        full_model = getattr(self, "_full_vision_model", None)
        if full_model is not None and hasattr(full_model, "gradient_checkpointing_enable"):
            full_model.gradient_checkpointing_enable()
            logger.info("SigLIP2wVAE: gradient checkpointing enabled")
        else:
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            checkpoint_fn = partial(checkpoint, use_reentrant=False)
            def _enable(module):
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    if getattr(module, "_gradient_checkpointing_func", None) is None:
                        module._gradient_checkpointing_func = checkpoint_fn
            self.model.apply(_enable)
            logger.info("SigLIP2wVAE: gradient checkpointing enabled (fallback)")

    def set_training_mode(self, mode: str = 'full', return_stats: bool = False):
        """Set encoder training mode."""
        self.requires_grad_(False)
        if mode == 'full':
            logger.info("SigLIP2wVAE: training full model")
            self.requires_grad_(True)
        elif mode == "no_sigliphead":
            logger.info("SigLIP2wVAE: siglip head no grad, siglip postnorm no grad")
            self.requires_grad_(True)
            if hasattr(self.model, 'head') and self.model.head is not None:
                for param in self.model.head.parameters():
                    param.requires_grad = False
            if hasattr(self.model, 'post_layernorm'):
                for param in self.model.post_layernorm.parameters():
                    param.requires_grad = False
        elif mode == "vae_only":
            logger.info("SigLIP2wVAE: training VAE heads only")
            self.requires_grad_(False)
            for module in [self.mu_head, self.logvar_head, self.pre_norm, self.post_norm]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        elif mode == 'frozen':
            pass  # Already frozen
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        if return_stats:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            return {'trainable_params': trainable_params, 'total_params': total_params}
        return None

    def get_last_layer(self):
        """Return the last layer for adaptive weight calculation."""
        if self.vae_enabled and self.logvar_head is not None:
            if hasattr(self.logvar_head, 'get_last_layer'):
                return self.logvar_head.get_last_layer()
            elif hasattr(self.logvar_head, 'up'):
                return self.logvar_head.up.weight
            elif hasattr(self.logvar_head, 'ffn'):
                return self.logvar_head.ffn[-1].weight
        return self.model.encoder.layers[-1].mlp.fc2.weight

    def _vae_sample(self, hidden_states: torch.Tensor):
        """Perform VAE sampling: pre_norm -> mu/logvar -> sampling."""
        # pre_norm before VAE head
        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)

        # Compute mu and logvar
        mu = self.mu_head(hidden_states)
        logvar = self.logvar_head(hidden_states)
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])

        # Reparameterization trick (training) or deterministic (eval)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar

    def forward(self, images, **kwargs):
        """
        Forward pass with VAE sampling.

        Args:
            images: (B, C, H, W) input images

        Returns:
            image_features: (B, N, D) latent features
            aux (optional): dict with "hidden_states_before_postnorm", "kl_loss", "mu", "logvar"
        """
        return_aux = kwargs.get("return_aux", False)

        # Encoder forward
        hidden_states = self.model.embeddings(images, interpolate_pos_encoding=True)
        encoder_outputs = self.model.encoder(inputs_embeds=hidden_states, output_hidden_states=True)
        pre_postnorm_hidden_states = encoder_outputs.last_hidden_state

        mu, logvar = None, None

        if self.vae_enabled:
            # VAE sampling: pre_norm -> VAE_head -> sampling
            z, mu, logvar = self._vae_sample(pre_postnorm_hidden_states)
        else:
            # VAE disabled, use encoder output directly
            z = pre_postnorm_hidden_states

        # post_norm: before decoder
        if self.post_norm is not None:
            image_features = self.post_norm(z)
        else:
            image_features = z

        # Prepare output
        if return_aux:
            aux = {"hidden_states_before_postnorm": pre_postnorm_hidden_states}

            # Compute KL loss only in training mode
            if self.vae_enabled and self.training and mu is not None and logvar is not None:
                kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                if self.kl_reduce == "mean":
                    kl_loss = kl_per_dim.mean()
                else:  # "sum_then_mean" (default)
                    kl_loss = kl_per_dim.sum(dim=-1).mean()
                aux["kl_loss"] = kl_loss
                aux["mu"] = mu
                aux["logvar"] = logvar

            return image_features, aux

        return image_features


@register_encoder()
class SigLIP2wTwoBackboneVAE(nn.Module):
    """Two backbones (semantic + recon). Recon tokens go through VAE sampling, then added to semantic tokens as image features. No normalization for now.

    Flow:
        semantic_tokens = semantic_backbone(images)
        recon_tokens = recon_backbone(images)
        z = VAE_sample(recon_tokens)  # pre_norm -> mu/logvar -> sampling
        [optional: z = recon_proj(z) if hidden sizes differ]
        image_features = semantic_tokens + z
    """

    def __init__(
        self,
        sementic_model_name: str = None,
        recon_model_name: str = None,
        num_tokens: int = 256,
        vae_enabled: bool = True,
        vae_nokl_enabled: bool = False,
        vae_head_adapter: str = "LinearAdapter",
        vae_head_adapter_kwargs: dict = None,
        pre_norm: bool = True,
        logvar_clamp: tuple = (-10.0, 10.0),
        vae_head_init: str = "default",
        kl_reduce: str = "sum_then_mean",  # "sum_then_mean": sum over dim then mean; "mean": global mean over all elements
        vae_sample_mode: str = "auto",  # "auto": sample during training, mu during eval; "sample": always sample; "mode": always use mu
        **kwargs,
    ):
        super().__init__()
        if not sementic_model_name or not recon_model_name:
            raise ValueError("SigLIP2wTwoBackboneVAE requires sementic_model_name and recon_model_name (in encoder_params)")
        logger.info(
            f"SigLIP2wTwoBackboneVAE init: sementic_model_name={sementic_model_name}, recon_model_name={recon_model_name}"
        )
        self.sementic_model_name = sementic_model_name
        self.recon_model_name = recon_model_name
        self.num_tokens = num_tokens
        self.vae_enabled = vae_enabled
        self.vae_nokl_enabled = vae_nokl_enabled
        self.logvar_clamp = logvar_clamp
        self.vae_head_init = vae_head_init
        if kl_reduce not in ("sum_then_mean", "mean"):
            raise ValueError(f"SigLIP2wTwoBackboneVAE: unknown kl_reduce='{kl_reduce}', choose 'sum_then_mean' or 'mean'")
        self.kl_reduce = kl_reduce
        logger.info(f"SigLIP2wTwoBackboneVAE: kl_reduce='{kl_reduce}'")
        if vae_sample_mode not in ("auto", "sample", "mode"):
            raise ValueError(f"SigLIP2wTwoBackboneVAE: unknown vae_sample_mode='{vae_sample_mode}', choose 'auto', 'sample' or 'mode'")
        self.vae_sample_mode = vae_sample_mode
        logger.info(f"SigLIP2wTwoBackboneVAE: vae_sample_mode='{vae_sample_mode}'")

        # Semantic backbone
        sementic_full = SiglipModel.from_pretrained(sementic_model_name)
        self.sementic_backbone = sementic_full.vision_model
        self.hidden_size = self.sementic_backbone.config.hidden_size
        self.patch_size = self.sementic_backbone.config.patch_size

        # Recon backbone
        recon_random_init = kwargs.get("recon_random_init", False)
        if recon_random_init:
            config = SiglipConfig.from_pretrained(recon_model_name)
            recon_full = SiglipVisionModel(config.vision_config)
            self.recon_backbone = recon_full.vision_model
            logger.warning("SigLIP2wTwoBackboneVAE: recon backbone random initialized")
        else:
            recon_full = SiglipModel.from_pretrained(recon_model_name)
            self.recon_backbone = recon_full.vision_model
        self.recon_hidden_size = self.recon_backbone.config.hidden_size

        # Project recon VAE output to semantic dim when sizes differ
        # if self.recon_hidden_size != self.hidden_size:
        #     self.recon_proj = nn.Linear(self.recon_hidden_size, self.hidden_size)
        #     nn.init.zeros_(self.recon_proj.weight)
        #     nn.init.zeros_(self.recon_proj.bias)
        #     logger.info(
        #         f"SigLIP2wTwoBackboneVAE: recon_proj {self.recon_hidden_size} -> {self.hidden_size}"
        #     )
        # else:
        #     self.recon_proj = None

        # Pre-norm before VAE head (on recon branch)
        vae_hidden = self.recon_hidden_size
        self.pre_norm = (
            nn.LayerNorm(vae_hidden, elementwise_affine=False) if pre_norm else None
        ) #这是vae的pre norm
        logger.info(
            f"SigLIP2wTwoBackboneVAE: pre_norm={'enabled' if self.pre_norm else 'disabled'}"
        )

        # VAE heads on recon branch
        if self.vae_enabled:
            adapter_kwargs = vae_head_adapter_kwargs or {}
            self.mu_head = build_adapter(
                vae_hidden, vae_head_adapter, out_dim=vae_hidden, **adapter_kwargs
            )
            self.logvar_head = build_adapter(
                vae_hidden, vae_head_adapter, out_dim=vae_hidden, **adapter_kwargs
            )
            logger.info(
                f"SigLIP2wTwoBackboneVAE: mu_head and logvar_head using {vae_head_adapter}"
            )
            self.vae_weight = float(kwargs.get("vae_weight", 1))
            self.noise_weight = float(kwargs.get("noise_weight", 1))
            self.mu_norm_threshold = float(kwargs.get("mu_norm_threshold", 0.0))
            if self.mu_norm_threshold > 0:
                logger.info(
                    f"SigLIP2wTwoBackboneVAE: mu_norm_threshold={self.mu_norm_threshold} "
                    f"(clamp vae_weight*||mu||_2 <= {self.mu_norm_threshold}*||semantic||_2)"
                )
            self._init_vae_heads(self.vae_head_init)
        elif self.vae_nokl_enabled:
            adapter_kwargs = vae_head_adapter_kwargs or {}
            self.mu_head = build_adapter(
                vae_hidden, vae_head_adapter, out_dim=vae_hidden, **adapter_kwargs
            )
            self.logvar_head = None
            logger.info(
                f"SigLIP2wTwoBackboneVAE: mu_head and logvar_head using {vae_head_adapter}"
            )
            self.vae_weight = float(kwargs.get("vae_weight", 1))
            self.noise_weight = float(kwargs.get("noise_weight", 1))
            logger.debug("no kl vae enable")
            self.clamp_mu = kwargs.get("clamp_mu", False)
            if self.clamp_mu:
                self.max_mu = float(kwargs.get("max_mu", 5.0))
                self.min_mu = float(kwargs.get("min_mu", -5.0))
                logger.info(f"SigLIP2wTwoBackboneVAE: clamp_mu enabled, range=[{self.min_mu}, {self.max_mu}]")
            self.wasserstein_enabled = kwargs.get("wasserstein_enabled", False)
            if self.wasserstein_enabled:
                self.wasserstein_reduce = kwargs.get("wasserstein_reduce", "token")  # "token" or "instance"
                if self.wasserstein_reduce not in ("token", "instance"):
                    raise ValueError(f"wasserstein_reduce must be 'token' or 'instance', got '{self.wasserstein_reduce}'")
                logger.info(f"SigLIP2wTwoBackboneVAE: wasserstein_enabled=True, reduce='{self.wasserstein_reduce}'")
            # self.mu_norm_threshold = float(kwargs.get("mu_norm_threshold", 0.0))
            # if self.mu_norm_threshold > 0:
            #     logger.info(
            #         f"SigLIP2wTwoBackboneVAE: mu_norm_threshold={self.mu_norm_threshold} "
            #         f"(clamp vae_weight*||mu||_2 <= {self.mu_norm_threshold}*||semantic||_2)"
            #     )
            # self._init_vae_heads(self.vae_head_init) 这里是希望
        else:
            self.mu_head = None
            self.logvar_head = None
            logger.info("SigLIP2wTwoBackboneVAE: VAE disabled")
            self.vae_weight = float(kwargs.get("vae_weight", 1))
            self.noise_weight = float(kwargs.get("noise_weight", 1))

        sementic_post_norm_type = kwargs.get("sementic_post_norm_type", "siglip_post_norm")
        if sementic_post_norm_type == "siglip_post_norm":
            self.sementic_post_layernorm = self.sementic_backbone.post_layernorm
        elif sementic_post_norm_type == "siglip_post_norm_recon":
            logger.info("use siglip_post_norm_recon")
            # 用recon分支的，而不是用semantic分支的，方便统一管理参数可调
            self.sementic_post_layernorm = self.recon_backbone.post_layernorm
        else:
            raise ValueError(f"Unknown post_norm_type: {sementic_post_norm_type}")

        use_semantic_head =  kwargs.get("use_semantic_head", False)
        if use_semantic_head:
            logger.info("use recon head as semantic_head")
            self.semantic_head = self.recon_backbone.head
        else:
            self.semantic_head = None

        #sementic scale norm
        self.sementic_scale_normalization = kwargs.get("sementic_scale_normalization", False)
        if self.sementic_scale_normalization and kwargs.get("norm_stat_path", None) is not None:
            stats = torch.load(kwargs.get("norm_stat_path"), map_location='cpu')
            self.sementic_latent_mean = stats.get('mean', None)
            self.sementic_latent_var = stats.get('var', None)

            if self.sementic_latent_mean.dim() == 3:
                # [ C H W] -> [N, C]
                self.sementic_latent_mean = self.sementic_latent_mean.view(self.sementic_latent_mean.shape[0], -1).transpose(0, 1)
            if self.sementic_latent_var.dim() == 3:
                self.sementic_latent_var = self.sementic_latent_var.view(self.sementic_latent_var.shape[0], -1).transpose(0, 1)


            self.eps = kwargs.get("eps", 1e-6)
            logger.debug(f"latent mean mean: {self.sementic_latent_mean.mean().item()}, latent mean std: {self.sementic_latent_mean.std().item()}")
            logger.debug(f"latent var mean: {self.sementic_latent_var.mean().item()}, latent var std: {self.sementic_latent_var.std().item()}")
            logger.warning(f"sementic_scale_normalization  is enabled, latent_mean={self.sementic_latent_mean}, latent_var={self.sementic_latent_var}, eps={self.eps}")


        use_semantic_proj = kwargs.get("use_semantic_proj", False)
        if use_semantic_proj:
            sementic_proj_cls = kwargs.get("sementic_proj_cls", "LinearAdapter")
            sementic_kw = kwargs.get("sementic_proj_kwargs")
            self.sementic_proj = build_adapter(self.sementic_backbone.config.hidden_size, cls_or_name=sementic_proj_cls, **sementic_kw)
        else:
            self.sementic_proj = None

        use_sementic_prenorm = kwargs.get("use_sementic_prenorm", False)
        if use_sementic_prenorm:
            logger.info("use semantic prenorm")
            self.sementic_prenorm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        else:
            self.sementic_prenorm = None

        self.merge_method = kwargs.get("merge_method", "add")
        logger.info(f"merge method {self.merge_method}")



        if kwargs.get("gradient_checkpointing", False):
            self.set_gradient_checkpointing()

    def _init_vae_heads(self, init_type: str):
        if init_type == "default":
            logger.info(
                "SigLIP2wTwoBackboneVAE: VAE heads using default PyTorch initialization"
            )
            return

        def _get_linear_layer(module):
            if hasattr(module, "linear"):
                return module.linear
            elif hasattr(module, "mlp"):
                last_idx = getattr(module, "_last_linear_idx", -1)
                if last_idx >= 0:
                    return module.mlp[last_idx]
            elif hasattr(module, "up"):
                return module.up
            return None

        mu_linear = _get_linear_layer(self.mu_head)
        logvar_linear = _get_linear_layer(self.logvar_head)

        if init_type == "zero":
            if mu_linear is not None:
                nn.init.zeros_(mu_linear.weight)
                if mu_linear.bias is not None:
                    nn.init.zeros_(mu_linear.bias)
            if logvar_linear is not None:
                nn.init.zeros_(logvar_linear.weight)
                if logvar_linear.bias is not None:
                    nn.init.zeros_(logvar_linear.bias)
            logger.info(
                "SigLIP2wTwoBackboneVAE: VAE heads initialized to zero (mu=0, logvar=0 -> z~N(0,1))"
            )
        elif init_type == "identity":
            if mu_linear is not None:
                if mu_linear.weight.shape[0] == mu_linear.weight.shape[1]:
                    nn.init.eye_(mu_linear.weight)
                else:
                    nn.init.normal_(mu_linear.weight, mean=0, std=0.02)
                if mu_linear.bias is not None:
                    nn.init.zeros_(mu_linear.bias)
            if logvar_linear is not None:
                nn.init.zeros_(logvar_linear.weight)
                if logvar_linear.bias is not None:
                    nn.init.zeros_(logvar_linear.bias)
            logger.info(
                "SigLIP2wTwoBackboneVAE: VAE heads initialized as identity (mu=input, logvar=0 -> z=input+noise)"
            )
        else:
            logger.warning(
                f"SigLIP2wTwoBackboneVAE: Unknown vae_head_init='{init_type}', using default"
            )

    def set_gradient_checkpointing(self):
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        checkpoint_fn = partial(checkpoint, use_reentrant=False)
        def _enable(module):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                if getattr(module, "_gradient_checkpointing_func", None) is None:
                    module._gradient_checkpointing_func = checkpoint_fn
        self.sementic_backbone.apply(_enable)
        self.recon_backbone.apply(_enable)
        logger.info("SigLIP2wTwoBackboneVAE: gradient checkpointing enabled")

    def set_training_mode(self, mode: str = "full", return_stats: bool = False):
        self.requires_grad_(False)
        if mode == "full":
            logger.info("SigLIP2wTwoBackboneVAE: training full model")
            self.requires_grad_(True)
        elif mode == "vae_only":
            logger.info("SigLIP2wTwoBackboneVAE: training VAE heads only")
            for module in [self.mu_head, self.logvar_head, self.pre_norm]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        elif mode == "recon_only":
            logger.info("SigLIP2wTwoBackboneVAE: training recon backbone + VAE only")
            for p in self.recon_backbone.parameters():
                p.requires_grad = True
            if hasattr(self.recon_backbone, "head") and self.recon_backbone.head is not None:
                for param in self.recon_backbone.head.parameters():
                    param.requires_grad = False
            if hasattr(self.recon_backbone, "post_layernorm"):
                for param in self.recon_backbone.post_layernorm.parameters():
                    param.requires_grad = False
            for module in [self.mu_head, self.logvar_head, self.pre_norm]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        elif mode == "recon_only_w_head_norm":
            logger.info("SigLIP2wTwoBackboneVAE: training recon backbone + VAE only+ recon backbone postnorm/head")
            for p in self.recon_backbone.parameters():
                p.requires_grad = True
            # if hasattr(self.recon_backbone, "head") and self.recon_backbone.head is not None:
            #     for param in self.recon_backbone.head.parameters():
            #         param.requires_grad = False
            # if hasattr(self.recon_backbone, "post_layernorm"):
            #     for param in self.recon_backbone.post_layernorm.parameters():
            #         param.requires_grad = False
            for module in [self.mu_head, self.logvar_head, self.pre_norm]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        elif mode == "recon_only_w_semantic_proj":
            logger.info("SigLIP2wTwoBackboneVAE: training recon backbone + VAE only with semantic proj")
            for p in self.recon_backbone.parameters():
                p.requires_grad = True
            if hasattr(self.recon_backbone, "head") and self.recon_backbone.head is not None:
                for param in self.recon_backbone.head.parameters():
                    param.requires_grad = False
            if hasattr(self.recon_backbone, "post_layernorm"):
                for param in self.recon_backbone.post_layernorm.parameters():
                    param.requires_grad = False
            for module in [self.mu_head, self.logvar_head, self.pre_norm, self.sementic_proj]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        elif mode == "frozen":
            pass
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        if return_stats:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable_params": trainable, "total_params": total}
        return None

    def get_last_layer(self):
        if self.vae_enabled and self.logvar_head is not None:
            if hasattr(self.logvar_head, "get_last_layer"):
                return self.logvar_head.get_last_layer()
            if hasattr(self.logvar_head, "up"):
                return self.logvar_head.up.weight
            if hasattr(self.logvar_head, "ffn"):
                return self.logvar_head.ffn[-1].weight
        return self.recon_backbone.encoder.layers[-1].mlp.fc2.weight

    def _get_tokens(self, backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
        try:
            hidden_states = backbone.embeddings(images, interpolate_pos_encoding=True)
        except TypeError:
            hidden_states = backbone.embeddings(images)
        encoder_out = backbone.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        )
        return encoder_out.last_hidden_state

    def _vae_sample(self, hidden_states: torch.Tensor, force_sample: bool = None, semantic_tokens_for_clamp: torch.Tensor = None):
        """Recon tokens -> pre_norm -> mu/logvar -> sampling -> z.

        Sampling behaviour is controlled by ``self.vae_sample_mode``:
          - ``"auto"``  : sample when training, use mu at eval (original behaviour).
          - ``"sample"``: always draw a random sample (even at eval time).
          - ``"mode"``  : always use mu, never sample.
        ``force_sample`` can override the above when explicitly provided:
          - ``True``  forces sampling regardless of mode/training state.
          - ``False`` forces mu regardless of mode/training state.
        ``semantic_tokens_for_clamp``: if provided and mu_norm_threshold > 0, clamp
          ||mu||_2 so that vae_weight * ||mu_i||_2 <= mu_norm_threshold * ||sem_i||_2.

        Returns (z, mu, logvar, did_sample) where ``did_sample`` indicates
        whether stochastic sampling was actually performed.
        """
        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)
        mu = self.mu_head(hidden_states)

        # --- mu norm clamp ---
        # 对每个 token 先算 D 维 L2 norm，再对 N 个 token 取平均，得到 per-image 的平均 token norm [B]
        if semantic_tokens_for_clamp is not None and self.mu_norm_threshold > 0:
            with torch.no_grad():
                mu_norm  = mu.norm(dim=-1)                          # [B,N]: mean over N of ||mu_n||_2
                sem_norm = semantic_tokens_for_clamp.norm(dim=-1).mean(dim=1)                 # [B]: mean over N of ||sem_n||_2
                max_mu   = self.mu_norm_threshold * sem_norm / (self.vae_weight + 1e-8)       # [B,N]
                scale    = (max_mu.unsqueeze(1) / mu_norm.clamp(min=1e-6)).clamp(max=1.0)                  # [B,N], <=1
            mu = mu * scale.unsqueeze(-1)  # scale 是 detach 的常数，梯度正常流过 mu
            logger.debug(
                f"mu norm clamp: mu_norm_before={mu_norm.max().item():.4f}, "
                f"sem_norm={sem_norm.mean().item():.4f}, "
                f"scale_mean={scale.mean().item():.4f}, scale_min={scale.min().item():.4f}"
            )

        logvar = self.logvar_head(hidden_states)
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])

        # Determine whether to sample
        if force_sample is not None:
            do_sample = force_sample
        elif self.vae_sample_mode == "sample":
            do_sample = True
        elif self.vae_sample_mode == "mode":
            do_sample = False
        else:  # "auto"
            do_sample = self.training

        if do_sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            logger.debug("vae no sampling, use mu")
            z = mu
        return z, mu, logvar, do_sample

    def _vae_nokl_sample(self, hidden_states: torch.Tensor, force_sample: bool = None, semantic_tokens_for_clamp: torch.Tensor = None):
        """Recon tokens -> pre_norm -> mu/logvar -> sampling -> z.

        Sampling behaviour is controlled by ``self.vae_sample_mode``:
          - ``"auto"``  : sample when training, use mu at eval (original behaviour).
          - ``"sample"``: always draw a random sample (even at eval time).
          - ``"mode"``  : always use mu, never sample.
        ``force_sample`` can override the above when explicitly provided:
          - ``True``  forces sampling regardless of mode/training state.
          - ``False`` forces mu regardless of mode/training state.
        ``semantic_tokens_for_clamp``: if provided and mu_norm_threshold > 0, clamp
          ||mu||_2 so that vae_weight * ||mu_i||_2 <= mu_norm_threshold * ||sem_i||_2.

        Returns (z, mu, logvar, did_sample) where ``did_sample`` indicates
        whether stochastic sampling was actually performed.
        """
        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)
        mu = self.mu_head(hidden_states)

        if self.clamp_mu:
            mu = torch.clamp(mu, self.min_mu, self.max_mu)
        z = mu

        wasserstein_loss = None
        if self.wasserstein_enabled and self.training:
            if self.wasserstein_reduce == "token":
                # per-token: 对每个token的D维做分位数对齐，shape (B, N, D)
                z_sorted, _ = z.sort(dim=-1)  # (B, N, D)
                D = z.shape[-1]
                quantiles = torch.linspace(0, 1, D, device=z.device).clamp(1e-6, 1 - 1e-6)
                target_quantiles = (torch.erfinv(2 * quantiles - 1) * (2 ** 0.5))  # (D,)
                wasserstein_loss = (z_sorted - target_quantiles).pow(2).mean()
            else:  # "instance"
                # per-instance: 对每张图的所有token展平后做分位数对齐，shape (B, N*D)
                B = z.shape[0]
                z_flat = z.reshape(B, -1)  # (B, N*D)
                z_sorted, _ = z_flat.sort(dim=-1)  # (B, N*D)
                ND = z_flat.shape[-1]
                quantiles = torch.linspace(0, 1, ND, device=z.device).clamp(1e-6, 1 - 1e-6)
                target_quantiles = (torch.erfinv(2 * quantiles - 1) * (2 ** 0.5))  # (N*D,)
                wasserstein_loss = (z_sorted - target_quantiles).pow(2).mean()

        return z, mu, wasserstein_loss, False

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        _, aux = self.forward(images, return_aux=True)
        sementic_hidden_states = aux["sementic_tokens"]
        if self.semantic_head:
            return self.semantic_head(sementic_hidden_states)
        else:
            return self.sementic_backbone.head(sementic_hidden_states)

    def _build_vae_dist(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        semantic_tokens_normed: torch.Tensor,
    ) -> dict:
        """Build the vae_dist dict to be stored in aux and (optionally) on disk.

        Both ``mu_full`` and ``logvar`` are kept in **norm space** (i.e. before
        sementic_scale denorm) so they can be serialised and later fed back into
        ``sample_latents`` which handles the denorm internally.

        mu_full = semantic_tokens_normed + mu_recon   (norm space)
        logvar  = logvar_recon                         (norm space)

        Returns dict with keys ``mu`` and ``logvar`` only – no closures, safe to
        serialise with ``torch.save``.
        """
        # mu_full: combine semantic (norm space) with the recon mu
        mu_recon = mu
        # if self.recon_proj is not None:
        #     mu_recon = self.recon_proj(mu_recon)
        mu_full = semantic_tokens_normed + mu_recon  # norm space
        return {"mu": mu_full, "logvar": logvar}

    def sample_latents(self, mu: torch.Tensor, logvar: torch.Tensor, mode: str = "mode") -> torch.Tensor:
        """Sample image_features from a stored vae_dist dict (loaded from disk or live).

        Both ``mu`` and ``logvar`` are expected to be in **norm space**
        (i.e. as produced by ``_build_vae_dist`` / stored on disk).

        Steps:
            1. Sample in norm space: z_norm = mu  (mode) or mu + std*eps  (standard)
            2. Denorm back to image_features space if sementic_scale_normalization is on.

        This is the public counterpart of the old ``sample_fn`` closure and can be
        called at training time on precomputed latents.
        """
        logvar_c = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])
        if mode == "mode":
            z_norm = mu
        else:  # "standard"
            std = torch.exp(0.5 * logvar_c)
            z_norm = mu + std * torch.randn_like(std)

        if self.sementic_scale_normalization and self.sementic_latent_mean is not None and self.sementic_latent_var is not None:
            latent_mean = self.sementic_latent_mean.to(z_norm.device)
            latent_var = self.sementic_latent_var.to(z_norm.device)
            return z_norm * torch.sqrt(latent_var + self.eps) + latent_mean
        return z_norm

    def merge_tokens(self, semntic_token, recon_token, force_merge_train: bool = False):
        batch_size, n ,d = semntic_token.shape
        device = semntic_token.device

        recon_token = self.vae_weight * recon_token
        logger.debug(f"recon token weight{self.vae_weight}")

        if not self.training and not force_merge_train:
            logger.debug("semantic add z")
            return semntic_token + recon_token, None
        if self.merge_method == "add":
            return semntic_token + recon_token, None
        elif self.merge_method == "bool":
            logger.debug("bool merge")
            t = torch.randint(0, 2, (batch_size,1, 1), device=device).float()
            mixed_token = (1 - t) * semntic_token + t * (semntic_token+recon_token)
            return mixed_token, t.view(batch_size)  # return t as cls_label [B]
        elif self.merge_method == "bool_noise":
            logger.debug(f"bool noise, noise weight{self.noise_weight}")
            # 随机选择：要么 semantic_token + randn(0,1)，要么 semantic_token + recon_token
            t = torch.randint(0, 2, (batch_size, 1, 1), device=device).float()
            noise = torch.randn_like(recon_token)
            mixed_token = semntic_token + (1 - t) * self.noise_weight * noise + t * recon_token
            return mixed_token, t.view(batch_size)  # return t as cls_label [B]
        else:
            raise NotImplementedError
    def output_head_forward(self, hidden_states, mean_first = True):
        assert self.semantic_head is not None, "output_head is not initialized"

        # hidden_states: (B, N, H, W) or (B, N, C), 是已经没有prefix tokens的输出
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
            logger.debug(f"hidden_states shape: {hidden_states.shape}")
        logger.debug("output head")
        return self.semantic_head(hidden_states) #[B N C]

    def forward(self, images: torch.Tensor, image_features_mode: str = None, **kwargs):
        """
        image_features = post_layernorm(denorm(semantic_tokens + VAE_sample(recon_tokens))).
        Optional aux: sementic_tokens, recon_z, kl_loss, mu, logvar, vae_dist.
        """
        return_aux = kwargs.get("return_aux", False)
        force_sample = kwargs.get("force_sample", None)
        force_merge_train = kwargs.get("force_merge_train", False)

        semantic_tokens = self._get_tokens(self.sementic_backbone, images)
        # logger.debug(f"semantic std before norm:{semantic_tokens.std().detach().item()}")

        if self.sementic_prenorm:
            logger.debug("semantic prenorm")
            semantic_tokens = self.sementic_prenorm(semantic_tokens)
        # logger.debug(f"semantic std:{semantic_tokens.std().detach().item()}")
        latent_mean, latent_var = None, None
        if self.sementic_scale_normalization:
            latent_mean = self.sementic_latent_mean.to(semantic_tokens.device)
            latent_var = self.sementic_latent_var.to(semantic_tokens.device)
            semantic_tokens = (semantic_tokens - latent_mean) / torch.sqrt(latent_var + self.eps)

        recon_tokens = self._get_tokens(self.recon_backbone, images)

        mu, logvar, did_sample = None, None, False
        if self.vae_enabled:
            z, mu, logvar, did_sample = self._vae_sample(
                recon_tokens,
                force_sample=force_sample,
                semantic_tokens_for_clamp=semantic_tokens if self.mu_norm_threshold > 0 else None,
            )
        elif self.vae_nokl_enabled:
            logger.debug("no kl vae sample")
            z, mu, wasserstein_loss, did_sample = self._vae_nokl_sample(
                recon_tokens,
                None,
                None,
            )
            logvar = None
        else:
            z = recon_tokens

        # if self.recon_proj is not None:
        #     z = self.recon_proj(z)
        # print(image_features_mode)
        # No normalization: image_features = semantic + z
        cls_label = None
        if image_features_mode == "original_sementic_tokens":
            # print("original_sementic_tokens")
            image_features = semantic_tokens
        elif image_features_mode == "original_sementic_tokens_noise":
            # print("original_sementic_tokens")
            # print(torch.randn_like(semantic_tokens))
            image_features = semantic_tokens + torch.randn_like(semantic_tokens)
        else:
            image_features, cls_label = self.merge_tokens(semantic_tokens, z, force_merge_train=force_merge_train)


        # Build semantic tokens for aux: semantic + mu (the deterministic center)
        # Use a separate variable to avoid polluting semantic_tokens used above
        semantic_tokens_for_aux = semantic_tokens + self.vae_weight * mu if mu is not None else semantic_tokens
        if self.sementic_scale_normalization:
            image_features = image_features * torch.sqrt(latent_var + self.eps) + latent_mean
            semantic_tokens_for_aux = semantic_tokens_for_aux * torch.sqrt(latent_var + self.eps) + latent_mean

        if self.sementic_proj is not None:
            # 做一个proj，能更柔和的进行对齐
            semantic_tokens_for_aux = self.sementic_proj(semantic_tokens_for_aux)

        sementic_post = self.sementic_post_layernorm(semantic_tokens_for_aux)


        if return_aux:
            aux = {
                "sementic_tokens": sementic_post,
                "recon_z": z,
                "sementic_z": semantic_tokens,
                "recon_z_weighted": self.vae_weight * z
            }
            if self.vae_nokl_enabled and wasserstein_loss is not None:
                aux["wasserstein_loss"] = wasserstein_loss
            if self.vae_enabled and mu is not None and logvar is not None:
                # Always expose mu/logvar so callers (e.g. precompute scripts) can
                # reconstruct VAEDistribution regardless of training/eval mode.
                aux["mu"] = mu
                aux["logvar"] = logvar
                # KL loss is only meaningful when stochastic sampling was performed.
                if did_sample:
                    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                    if self.kl_reduce == "mean":
                        kl_loss = kl_per_dim.mean()
                    else:  # "sum_then_mean" (default)
                        kl_loss = kl_per_dim.sum(dim=-1).mean()
                    aux["kl_loss"] = kl_loss

                # Build a serialisable vae_dist dict for mode-C precompute.
                # mu_full and logvar are both in norm space.
                # Only when not in "original_sementic_tokens" mode (which doesn't use VAE).
                # if image_features_mode != "original_sementic_tokens":
                aux["vae_dist"] = self._build_vae_dist(
                        mu=mu,
                        logvar=logvar,
                        semantic_tokens_normed=semantic_tokens,  # already normalised above
                    )

            if cls_label is not None:
                aux["cls_label"] = cls_label
            return image_features, aux
        return image_features


