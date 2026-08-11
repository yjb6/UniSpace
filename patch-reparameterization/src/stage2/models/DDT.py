import logging
from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift,
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate,
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x),
                        shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(
            num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(
            patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        s_input_size = input_size
        s_patch_size = self.s_patch_size
        x_input_size = input_size
        x_patch_size = self.x_patch_size
        x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size
        self.x_embedder = PatchEmbed(
            x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True)
        self.s_embedder = PatchEmbed(
            s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True)
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, self.encoder_hidden_size, class_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder.num_patches))
            # print(f"enc_half_head_dim: {enc_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            # print(f"dec_half_head_dim: {dec_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                              mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              ) for i in range(self.num_blocks)
        ])
        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def encoder_forward(self, x, t, y):
        """Encoder part — safe under bf16 autocast, can be torch.compiled."""
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = nn.functional.silu(t + y)
        s = self.s_embedder(x)
        if self.use_pos_embed:
            s = s + self.pos_embed
        for i in range(self.num_encoder_blocks):
            s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
        t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
        s = nn.functional.silu(t + s)
        return s

    def decoder_forward(self, x, s):
        """Decoder part — runs in fp32 to prevent bf16 accumulation errors in DDT modulation."""
        with torch.cuda.amp.autocast(enabled=False):
            s = self.s_projector(s.float())
            x = self.x_embedder(x.float())
            if self.use_pos_embed and self.x_pos_embed is not None:
                x = x + self.x_pos_embed
            for i in range(self.num_encoder_blocks, self.num_blocks):
                x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
            x = self.final_layer(x, s)
            x = self.unpatchify(x)
        return x

    def forward(self, x, t, y, s=None, mask=None):
        if s is None:
            s = self.encoder_forward(x, t, y)
        return self.decoder_forward(x, s)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:,
                              :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps


class DiTwDDTHeadDualHead(DiTwDDTHead):
    """DDT with dual decoder heads: one for semantic, one for recon.

    Shared encoder (28 blocks) → two independent decoder branches:
      - sem branch: s_projector_sem → x_embedder_sem → decoder_sem_blocks → final_layer_sem → sem_channels
      - recon branch: s_projector_recon → x_embedder_recon → decoder_recon_blocks → final_layer_recon → recon_channels
    Outputs are concatenated back to in_channels.
    """

    def __init__(
            self,
            sem_channels: int = 768,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
    ):
        # Initialize parent (builds shared encoder + original decoder)
        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob,
            num_classes=num_classes, use_qknorm=use_qknorm, use_swiglu=use_swiglu,
            use_rope=use_rope, use_rmsnorm=use_rmsnorm, wo_shift=wo_shift,
            use_pos_embed=use_pos_embed,
        )
        recon_channels = in_channels - sem_channels
        assert recon_channels > 0, f"sem_channels={sem_channels} >= in_channels={in_channels}"
        self.sem_channels = sem_channels
        self.recon_channels = recon_channels

        p = self.x_patch_size
        dec_num_heads = self.num_heads[1]
        _mlp_ratio = mlp_ratio

        # Input size for x_embedder (spatial dim before patchify)
        _x_input_size = int(sqrt(self.x_embedder.num_patches)) * self.x_patch_size

        # --- Semantic decoder branch ---
        sem_ch_per_token = sem_channels * p * p
        self.x_embedder_sem = PatchEmbed(
            _x_input_size, self.x_patch_size, sem_ch_per_token, self.decoder_hidden_size, bias=True)
        self.s_projector_sem = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.decoder_blocks_sem = nn.ModuleList([
            LightningDDTBlock(self.decoder_hidden_size, dec_num_heads,
                              mlp_ratio=_mlp_ratio,
                              use_qknorm=use_qknorm, use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu, wo_shift=wo_shift)
            for _ in range(self.num_decoder_blocks)
        ])
        self.final_layer_sem = DDTFinalLayer(
            self.decoder_hidden_size, 1, sem_ch_per_token, use_rmsnorm=use_rmsnorm)

        # --- Recon decoder branch ---
        recon_ch_per_token = recon_channels * p * p
        self.x_embedder_recon = PatchEmbed(
            _x_input_size, self.x_patch_size, recon_ch_per_token, self.decoder_hidden_size, bias=True)
        self.s_projector_recon = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.decoder_blocks_recon = nn.ModuleList([
            LightningDDTBlock(self.decoder_hidden_size, dec_num_heads,
                              mlp_ratio=_mlp_ratio,
                              use_qknorm=use_qknorm, use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu, wo_shift=wo_shift)
            for _ in range(self.num_decoder_blocks)
        ])
        self.final_layer_recon = DDTFinalLayer(
            self.decoder_hidden_size, 1, recon_ch_per_token, use_rmsnorm=use_rmsnorm)

        # RoPE for new decoder branches (shared same config as original dec_feat_rope)
        if self.use_rope:
            self.dec_feat_rope_sem = self.dec_feat_rope
            self.dec_feat_rope_recon = self.dec_feat_rope

        # Remove parent's original decoder modules (unused, would cause DDP errors)
        del self.x_embedder
        del self.s_projector
        del self.final_layer
        # Remove parent's original decoder blocks (blocks[num_encoder_blocks:])
        original_blocks = list(self.blocks)
        self.blocks = nn.ModuleList(original_blocks[:self.num_encoder_blocks])

        # Initialize new modules
        self._init_dual_head_weights()

        logger.info(f"DiTwDDTHeadDualHead: sem_channels={sem_channels}, recon_channels={recon_channels}, "
                    f"decoder_blocks={self.num_decoder_blocks} x2, decoder_hidden={self.decoder_hidden_size}")

    def _init_dual_head_weights(self):
        """Initialize weights for the dual decoder branches."""
        for embedder in [self.x_embedder_sem, self.x_embedder_recon]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        for blocks in [self.decoder_blocks_sem, self.decoder_blocks_recon]:
            for block in blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for fl in [self.final_layer_sem, self.final_layer_recon]:
            nn.init.constant_(fl.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(fl.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(fl.linear.weight, 0)
            nn.init.constant_(fl.linear.bias, 0)

    def _unpatchify_branch(self, x, channels):
        """Unpatchify for a branch with specific number of channels."""
        c = channels * self.x_patch_size * self.x_patch_size
        p = self.x_patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def decoder_forward(self, x, s):
        """Dual decoder: sem and recon branches run independently, outputs concatenated."""
        with torch.cuda.amp.autocast(enabled=False):
            # Split input x along channel dim: [B, C, H, W] → sem [B, sem_ch, H, W] + recon [B, recon_ch, H, W]
            x_float = x.float()
            x_sem_input = x_float[:, :self.sem_channels, :, :]
            x_recon_input = x_float[:, self.sem_channels:, :, :]

            s_float = s.float()

            # --- Semantic branch ---
            s_sem = self.s_projector_sem(s_float)
            h_sem = self.x_embedder_sem(x_sem_input)
            rope_sem = self.dec_feat_rope_sem if self.use_rope else None
            for block in self.decoder_blocks_sem:
                h_sem = block(h_sem, s_sem, feat_rope=rope_sem)
            h_sem = self.final_layer_sem(h_sem, s_sem)
            out_sem = self._unpatchify_branch(h_sem, self.sem_channels)  # [B, sem_ch, H, W]

            # --- Recon branch ---
            s_recon = self.s_projector_recon(s_float)
            h_recon = self.x_embedder_recon(x_recon_input)
            rope_recon = self.dec_feat_rope_recon if self.use_rope else None
            for block in self.decoder_blocks_recon:
                h_recon = block(h_recon, s_recon, feat_rope=rope_recon)
            h_recon = self.final_layer_recon(h_recon, s_recon)
            out_recon = self._unpatchify_branch(h_recon, self.recon_channels)  # [B, recon_ch, H, W]

            # Concatenate: [B, sem_ch + recon_ch, H, W] = [B, in_channels, H, W]
            out = torch.cat([out_sem, out_recon], dim=1)
        return out


class PartialRoPE(nn.Module):
    """
    A RoPE wrapper that only applies rotation to the last `num_spatial` tokens
    in the sequence, leaving the leading `num_globaltoken` tokens unchanged.

    The underlying `feat_rope` (VisionRotaryEmbeddingFast) expects input of shape
    [B, num_heads, num_spatial, head_dim] and returns the same shape with RoPE applied.
    This wrapper slices q/k, applies RoPE to the spatial slice, then stitches back.

    Args:
        feat_rope:        a VisionRotaryEmbeddingFast instance sized for num_spatial tokens.
        num_globaltoken:  number of leading global tokens to skip.
    """
    def __init__(self, feat_rope, num_globaltoken: int):
        super().__init__()
        self.feat_rope = feat_rope
        self.num_globaltoken = num_globaltoken

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_heads, N, head_dim]  where N = num_globaltoken + num_spatial
        Returns:
            x: same shape, with RoPE applied only to x[:, :, num_globaltoken:, :]
        """
        g = self.num_globaltoken
        x_global  = x[:, :, :g, :]   # (B, H, num_globaltoken, D)  — untouched
        x_spatial = x[:, :, g:, :]   # (B, H, num_spatial,     D)  — apply RoPE
        x_spatial = self.feat_rope(x_spatial)
        return torch.cat([x_global, x_spatial], dim=2)


class DiTwDDTHeadTokenInput(nn.Module):
    """
    A variant of DiTwDDTHead that accepts pre-tokenized input x of shape [B, N, D],
    where N = num_globaltoken + H * W (global tokens come first).

    Key differences from DiTwDDTHead:
    - Input x is already tokenized: no PatchEmbed needed; a linear projector maps D -> hidden_size.
    - Position encoding:
        * spatial tokens (H*W) use fixed 2D sin-cos positional embedding.
        * global tokens use learnable positional embedding.
    - Encoder (DDT-style): all N tokens pass through encoder blocks together with AdaLN condition c.
      After encoding, timestep t is broadcast and fused into the token sequence as the condition s.
    - Decoder: all N tokens are denoised conditioned on s from the encoder via AdaLN.
    - Output: [B, N, D], same shape as input (both global and spatial tokens are denoised).
    """

    def __init__(
            self,
            in_channels: int = 768,         # D: input/output token dimension
            hw_size: int = 16,              # spatial grid side length, num spatial tokens = hw_size^2
            num_globaltoken: int = 8,       # number of global tokens (prepended before spatial tokens)
            hidden_size: Union[list, int] = [1152, 2048],  # [encoder_hidden, decoder_hidden]
            depth: list = [28, 2],          # [num_encoder_blocks, num_decoder_blocks]
            num_heads: Union[list, int] = [16, 16],
            mlp_ratio: float = 4.0,
            class_dropout_prob: float = 0.1,
            num_classes: int = 1000,
            use_qknorm: bool = False,
            use_swiglu: bool = True,
            use_rope: bool = False,         # RoPE is disabled by default for mixed token sequences
            use_rmsnorm: bool = True,
            wo_shift: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hw_size = hw_size
        self.num_spatial = hw_size * hw_size
        self.num_globaltoken = num_globaltoken
        self.num_tokens = num_globaltoken + self.num_spatial  # total N

        # hidden sizes
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size, hidden_size]
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]

        # num heads
        if isinstance(num_heads, int):
            num_heads = [num_heads, num_heads]
        self.num_heads = list(num_heads)
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]

        # block depths
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope

        # ---------- input projectors ----------
        # project D -> encoder_hidden_size for the encoder stage
        self.enc_input_proj = nn.Linear(in_channels, self.encoder_hidden_size, bias=True)
        # project D -> decoder_hidden_size for the decoder stage
        self.dec_input_proj = nn.Linear(in_channels, self.decoder_hidden_size, bias=True)

        # ---------- position embeddings ----------
        # spatial tokens: fixed 2D sin-cos, shape (1, num_spatial, encoder_hidden_size)
        self.enc_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial, self.encoder_hidden_size), requires_grad=False)
        self.dec_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial, self.decoder_hidden_size), requires_grad=False)

        # global tokens: learnable, shape (1, num_globaltoken, hidden_size)
        self.enc_global_pos_embed = nn.Parameter(
            torch.zeros(1, num_globaltoken, self.encoder_hidden_size), requires_grad=True)
        self.dec_global_pos_embed = nn.Parameter(
            torch.zeros(1, num_globaltoken, self.decoder_hidden_size), requires_grad=True)

        # ---------- condition embedders ----------
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob)

        # project encoder hidden -> decoder hidden for condition s
        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )

        # ---------- RoPE (optional, only applied to spatial tokens) ----------
        # global tokens are skipped; PartialRoPE wraps VisionRotaryEmbeddingFast
        # so that only the trailing num_spatial positions are rotated.
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            self.enc_feat_rope = PartialRoPE(
                feat_rope=VisionRotaryEmbeddingFast(
                    dim=enc_half_head_dim,
                    pt_seq_len=hw_size,
                ),
                num_globaltoken=num_globaltoken,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            self.dec_feat_rope = PartialRoPE(
                feat_rope=VisionRotaryEmbeddingFast(
                    dim=dec_half_head_dim,
                    pt_seq_len=hw_size,
                ),
                num_globaltoken=num_globaltoken,
            )
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        # ---------- transformer blocks ----------
        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            ) for i in range(self.num_blocks)
        ])

        # ---------- final layer ----------
        # outputs in_channels per token (no patch expansion)
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, in_channels, use_rmsnorm=use_rmsnorm)

        # ---------- output projector ----------
        # map decoder_hidden_size back to in_channels
        # (DDTFinalLayer already does linear(hidden -> 1*1*in_channels), so this is identity-like)
        # We keep it explicit for clarity; DDTFinalLayer.linear outputs in_channels directly.

        self.initialize_weights()

    # ------------------------------------------------------------------
    def initialize_weights(self):
        # sin-cos 2D pos embed for spatial tokens (encoder)
        enc_pos = get_2d_sincos_pos_embed(self.encoder_hidden_size, self.hw_size)
        self.enc_spatial_pos_embed.data.copy_(
            torch.from_numpy(enc_pos).float().unsqueeze(0))

        # sin-cos 2D pos embed for spatial tokens (decoder)
        dec_pos = get_2d_sincos_pos_embed(self.decoder_hidden_size, self.hw_size)
        self.dec_spatial_pos_embed.data.copy_(
            torch.from_numpy(dec_pos).float().unsqueeze(0))

        # learnable global pos embed: small normal init
        nn.init.normal_(self.enc_global_pos_embed, std=0.02)
        nn.init.normal_(self.dec_global_pos_embed, std=0.02)

        # input projectors
        nn.init.xavier_uniform_(self.enc_input_proj.weight)
        nn.init.constant_(self.enc_input_proj.bias, 0)
        nn.init.xavier_uniform_(self.dec_input_proj.weight)
        nn.init.constant_(self.dec_input_proj.bias, 0)

        # label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ------------------------------------------------------------------
    def _build_pos_embed(self, hidden_size, global_pos_embed, spatial_pos_embed):
        """
        Concatenate [global_pos_embed, spatial_pos_embed] along dim=1.
        global_pos_embed: (1, num_globaltoken, hidden_size)  — learnable
        spatial_pos_embed: (1, num_spatial, hidden_size)     — fixed sin-cos
        Returns: (1, num_tokens, hidden_size)
        """
        return torch.cat([global_pos_embed, spatial_pos_embed], dim=1)

    # ------------------------------------------------------------------
    def forward(self, x, t, y, s=None):
        """
        Args:
            x: [B, N, D]  noisy token sequence, N = num_globaltoken + H*W
            t: [B]        diffusion timestep
            y: [B]        class label (int)
            s: optional pre-computed encoder condition [B, N, encoder_hidden_size].
               If provided, the encoder stage is skipped.
        Returns:
            out: [B, N, D]  denoised token sequence (same shape as input)
        """
        B, N, D = x.shape
        assert N == self.num_tokens, (
            f"Expected N={self.num_tokens} (num_globaltoken={self.num_globaltoken} + "
            f"num_spatial={self.num_spatial}), got {N}"
        )

        # ---- condition embeddings ----
        t_emb = self.t_embedder(t)                       # (B, encoder_hidden_size)
        y_emb = self.y_embedder(y, self.training)        # (B, encoder_hidden_size)
        c = nn.functional.silu(t_emb + y_emb)           # (B, encoder_hidden_size)

        # ---- encoder stage ----
        if s is None:
            # project input to encoder hidden size
            xe = self.enc_input_proj(x)                  # (B, N, encoder_hidden_size)

            # add position embeddings:
            #   global tokens (front): learnable enc_global_pos_embed
            #   spatial tokens (back): fixed sin-cos enc_spatial_pos_embed
            pos = self._build_pos_embed(
                self.encoder_hidden_size,
                self.enc_global_pos_embed,               # (1, num_globaltoken, enc_h)
                self.enc_spatial_pos_embed,              # (1, num_spatial,     enc_h)
            )  # (1, N, encoder_hidden_size)
            xe = xe + pos

            # pass through encoder blocks (self-attention conditioned on c)
            # PartialRoPE ensures only spatial tokens (trailing hw_size^2) are rotated
            for i in range(self.num_encoder_blocks):
                xe = self.blocks[i](xe, c, feat_rope=self.enc_feat_rope)

            # fuse timestep into encoder output to form condition s
            t_broadcast = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, enc_h)
            s = nn.functional.silu(t_broadcast + xe)             # (B, N, encoder_hidden_size)

        # project condition s to decoder hidden size
        s = self.s_projector(s)                          # (B, N, decoder_hidden_size)

        # ---- decoder stage ----
        # project input to decoder hidden size
        xd = self.dec_input_proj(x)                      # (B, N, decoder_hidden_size)

        # add decoder position embeddings
        dec_pos = self._build_pos_embed(
            self.decoder_hidden_size,
            self.dec_global_pos_embed,                   # (1, num_globaltoken, dec_h)
            self.dec_spatial_pos_embed,                  # (1, num_spatial,     dec_h)
        )  # (1, N, decoder_hidden_size)
        xd = xd + dec_pos

        # pass through decoder blocks (conditioned on s)
        # PartialRoPE ensures only spatial tokens (trailing hw_size^2) are rotated
        for i in range(self.num_encoder_blocks, self.num_blocks):
            xd = self.blocks[i](xd, s, feat_rope=self.dec_feat_rope)

        # final layer: outputs (B, N, in_channels)
        out = self.final_layer(xd, s)                    # (B, N, in_channels)
        return out

    # ------------------------------------------------------------------
    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Classifier-free guidance forward pass.
        Assumes x is already doubled: first half conditional, second half unconditional.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)         # (2B, N, D)

        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max
        t_half = t[: len(t) // 2]
        half_eps = torch.where(
            ((t_half >= guid_t_min) & (t_half <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=-1)

    # ------------------------------------------------------------------
    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Auto-guidance forward pass using an additional (weaker) model.
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)

        eps = model_out[:, :, :self.in_channels]
        ag_eps = ag_model_out[:, :, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps),
            eps,
        )
        return eps
