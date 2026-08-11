# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device, mask_type="flex"):
    """
    创建稀疏注意力掩码，支持flex_attention和SDPA两种格式

    Args:
        document_lens: 文档长度列表，每个文档是一个独立的sample
        split_lens: 分割长度列表
        attn_modes: 注意力模式列表 ['causal', 'full', 'noise']
        device: 设备
        mask_type: 掩码类型，'flex' 或 'sdpa'

    Returns:
        根据mask_type返回相应格式的掩码
    """
    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    if mask_type == "flex":
        # flex_attention 的 create_block_mask 会将 Q_LEN/KV_LEN 向上取整到 BLOCK_SIZE 的倍数,
        # 导致 vmap 生成的索引可能超出张量长度。
        # 方案: 在闭包中用 clamp 限制索引, 并让越界位置的 document_id 不等于任何有效值 (sentinel=-1),
        # 从而让 sample_mask 自动屏蔽越界区域。
        seq_len = full_and_noise_seq_id.shape[0]
        # 给 document_id 追加一个 sentinel=0 (不等于任何有效 document_id >= 1)
        _document_id = torch.cat([document_id, torch.zeros(1, device=device)])
        # 给 full_and_noise_seq_id 和 noise_seq_id 追加 sentinel=-1 (不会匹配且 < 0)
        _full_and_noise_seq_id = torch.cat([full_and_noise_seq_id, torch.full((1,), -1, device=device)])
        _noise_seq_id = torch.cat([noise_seq_id, torch.full((1,), -1, device=device)])

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def full_and_noise_mask(b, h, q_idx, kv_idx):
            q_safe = torch.clamp(q_idx, max=seq_len)
            kv_safe = torch.clamp(kv_idx, max=seq_len)
            return (_full_and_noise_seq_id[q_safe] == _full_and_noise_seq_id[kv_safe]) & (_full_and_noise_seq_id[q_safe] >= 0)

        def remove_noise_mask(b, h, q_idx, kv_idx):
            q_safe = torch.clamp(q_idx, max=seq_len)
            kv_safe = torch.clamp(kv_idx, max=seq_len)
            return (~((_noise_seq_id[kv_safe] >= 0) & (_noise_seq_id[q_safe] != _noise_seq_id[kv_safe])))

        def sample_mask(b, h, q_idx, kv_idx):
            q_safe = torch.clamp(q_idx, max=seq_len)
            kv_safe = torch.clamp(kv_idx, max=seq_len)
            return _document_id[q_safe] == _document_id[kv_safe]

        # 返回flex_attention格式的掩码
        return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)
    elif mask_type == "sdpa":
        # 转换为SDPA格式的掩码
        total_seq_len = sum(document_lens)

        # 创建索引矩阵
        q_idx = torch.arange(total_seq_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(total_seq_len, device=device).unsqueeze(0)

        # 计算 sample_mask: document_id[q] == document_id[kv]
        # 不同sample之间完全隔离
        sample = document_id[q_idx] == document_id[kv_idx]

        # 计算 causal_mask: q_idx >= kv_idx
        causal = q_idx >= kv_idx

        # 计算 full_and_noise_mask: 相同的 full_and_noise_seq_id 且 >= 0
        # 注意：这个掩码只在同一个 sample 内有效
        full_and_noise = (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

        # 计算 or_masks(causal, full_and_noise_mask)
        # 这个结果需要与 sample_mask 做 AND，确保不同 sample 之间隔离
        causal_or_full_noise = causal | full_and_noise

        # 计算 remove_noise_mask: 排除 noise 段对其他段的影响
        # ~(noise_seq_id[kv_idx] >= 0 & noise_seq_id[q_idx] != noise_seq_id[kv_idx])
        remove_noise = ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

        # 最终掩码: and_masks(causal_or_full_noise, remove_noise, sample_mask)
        # 关键：sample_mask 必须首先满足，确保不同 sample 之间隔离
        mask = sample & causal_or_full_noise & remove_noise

        # 直接返回 bool 掩码: True=可见, False=屏蔽
        # NPU 的 FlashAttentionScore 算子要求 bool mask，float mask 会退化为小算子拼接
        # GPU 的 SDPA 同样支持 bool mask，行为一致
        # 扩展为 SDPA 期望的形状: (1, 1, seq_len, seq_len)
        sdpa_mask = mask.unsqueeze(0).unsqueeze(0)

        return sdpa_mask
    else:
        raise ValueError(f"不支持的mask_type: {mask_type}")


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    # 为 SDPA 添加 batch 和 heads 维度
    # SDPA 期望形状: (batch_size, num_heads, query_len, key_len)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


MULTI_ASPECT_RATIO_1024 = {
    '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.]
}


MULTI_ASPECT_RATIO_512 = {
    # 从 MULTI_ASPECT_RATIO_256 的 H/W 各 ×2 生成，宽高比完全对齐
    # --- extended: ratio 0.25~0.48, symmetric with 2.09~4.0 ---
    '0.25': [256.0, 1024.0], '0.32': [288.0, 896.0], '0.35': [288.0, 832.0],
    '0.4': [320.0, 800.0], '0.42': [320.0, 768.0], '0.48': [352.0, 736.0],
    # --- original: ratio 0.5~2.0 ---
    '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0],
    # --- extended: ratio 2.09~4.0, symmetric with 0.25~0.48 ---
    '2.09': [736.0, 352.0], '2.4': [768.0, 320.0], '2.5': [800.0, 320.0],
    '2.89': [832.0, 288.0], '3.11': [896.0, 288.0], '4.0': [1024.0, 256.0],
}


MULTI_ASPECT_RATIO_384 = {
    # --- extended: ratio 0.25~0.49, symmetric with 2.06~4.0 ---
    '0.25': [192.0, 768.0], '0.33': [224.0, 672.0], '0.34': [224.0, 656.0],
    '0.4': [240.0, 608.0], '0.43': [256.0, 592.0], '0.49': [272.0, 560.0],
    # --- original: ratio 0.5~2.0 ---
    '0.5': [272.0, 544.0], '0.52': [272.0, 528.0],
    '0.56': [288.0, 512.0], '0.6': [304.0, 496.0], '0.69': [320.0, 464.0], '0.71': [320.0, 448.0],
    '0.78': [336.0, 432.0], '0.81': [352.0, 432.0], '0.88': [368.0, 416.0], '0.92': [368.0, 400.0],
    '1.0': [384.0, 384.0], '1.09': [400.0, 368.0], '1.13': [416.0, 368.0], '1.18': [416.0, 352.0],
    '1.29': [432.0, 336.0], '1.4': [448.0, 320.0], '1.45': [464.0, 320.0], '1.63': [496.0, 304.0],
    '1.78': [512.0, 288.0], '2.0': [544.0, 272.0],
    # --- extended: ratio 2.06~4.0, symmetric with 0.25~0.49 ---
    '2.06': [560.0, 272.0], '2.47': [592.0, 240.0], '2.53': [608.0, 240.0],
    '2.93': [656.0, 224.0], '3.0': [672.0, 224.0], '4.0': [768.0, 192.0],
}


MULTI_ASPECT_RATIO_256 = {
    # --- extended: ratio 0.25~0.48, symmetric with 2.09~4.0 ---
    '0.25': [128.0, 512.0], '0.32': [144.0, 448.0], '0.35': [144.0, 416.0],
    '0.4': [160.0, 400.0], '0.42': [160.0, 384.0], '0.48': [176.0, 368.0],
    # --- original: ratio 0.5~2.0 ---
    '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
    '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
    '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
    '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
    '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
    '1.75': [336.0, 192.0], '2.0': [352.0, 176.0],
    # --- extended: ratio 2.09~4.0, symmetric with 0.25~0.48 ---
    '2.09': [368.0, 176.0], '2.4': [384.0, 160.0], '2.5': [400.0, 160.0],
    '2.89': [416.0, 144.0], '3.11': [448.0, 144.0], '4.0': [512.0, 128.0],
}

# mar_256 with all H/W rounded to multiples of 32
# Required by Qwen3Unified encoder (spatial_merge_size=2, patch_size=16 → H/W must be 32x)
MULTI_ASPECT_RATIO_256_S32 = {
    # All H and W are multiples of 32
    '0.25': [128.0, 512.0], '0.32': [160.0, 480.0], '0.35': [160.0, 448.0],
    '0.4':  [160.0, 384.0], '0.42': [160.0, 384.0], '0.48': [192.0, 384.0],
    '0.5':  [192.0, 384.0], '0.52': [192.0, 352.0],
    '0.57': [192.0, 320.0], '0.6':  [192.0, 320.0], '0.68': [224.0, 320.0], '0.72': [224.0, 320.0],
    '0.78': [224.0, 288.0], '0.82': [224.0, 256.0], '0.88': [256.0, 288.0], '0.94': [256.0, 256.0],
    '1.0':  [256.0, 256.0], '1.07': [256.0, 224.0], '1.13': [256.0, 224.0], '1.21': [288.0, 224.0],
    '1.29': [288.0, 224.0], '1.38': [288.0, 192.0], '1.46': [320.0, 224.0], '1.67': [320.0, 192.0],
    '1.75': [320.0, 192.0], '2.0':  [384.0, 192.0],
    '2.09': [384.0, 192.0], '2.4':  [384.0, 160.0], '2.5':  [384.0, 160.0],
    '2.89': [384.0, 128.0], '3.11': [448.0, 128.0], '4.0':  [512.0, 128.0],
}

MULTI_RESOLUTION_MAP = {
    'mar_256': MULTI_ASPECT_RATIO_256,
    'mar_256_s32': MULTI_ASPECT_RATIO_256_S32,
    'mar_384': MULTI_ASPECT_RATIO_384,
    'mar_512': MULTI_ASPECT_RATIO_512,
    'mar_1024': MULTI_ASPECT_RATIO_1024,
}

def create_sdpa_attention_mask(split_lens, attn_modes, device="cpu", num_heads=1):
    """
    为 SDPA 创建注意力掩码

    Args:
        split_lens: 每个分割的长度列表
        attn_modes: 每个分割的注意力模式 ['causal', 'full', 'noise']
        device: 设备
        num_heads: 注意力头数

    Returns:
        attention_mask: (1, num_heads, seq_len, seq_len) 格式的掩码
    """
    sample_len = sum(split_lens)

    # 创建基础掩码
    base_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            # 因果掩码：只能看到当前及之前的token
            base_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            base_mask[csum:csum + s, :csum] = 1  # 可以看到之前分割的所有token
        elif attn_mode == "full":
            # 全注意力：可以看到当前分割的所有token
            base_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            base_mask[csum:csum + s, :csum] = 1  # 可以看到之前分割的所有token
        # noise模式在下面单独处理
        csum += s

    # 处理noise模式
    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            # noise模式：只能看到当前分割的token，其他token看不到
            base_mask[:, csum : csum + s] = 0
            base_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    # 转换为SDPA格式：0表示可见，-inf表示不可见
    attention_mask = torch.zeros_like(base_mask, dtype=torch.float).masked_fill_(
        ~base_mask, float("-inf")
    )

    # 扩展为SDPA期望的形状: (batch_size, num_heads, query_len, key_len)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # 如果需要多个注意力头，重复掩码
    if num_heads > 1:
        attention_mask = attention_mask.expand(1, num_heads, sample_len, sample_len)

    return attention_mask


def create_sdpa_causal_mask(seq_len, device="cpu", num_heads=1):
    """
    创建SDPA因果注意力掩码

    Args:
        seq_len: 序列长度
        device: 设备
        num_heads: 注意力头数

    Returns:
        causal_mask: (1, num_heads, seq_len, seq_len) 格式的因果掩码
    """
    # 创建下三角矩阵
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # 转换为SDPA格式：0表示可见，-inf表示不可见
    causal_mask = torch.zeros_like(causal_mask, dtype=torch.float).masked_fill_(
        causal_mask == 0, float("-inf")
    )

    # 扩展为SDPA期望的形状
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    if num_heads > 1:
        causal_mask = causal_mask.expand(1, num_heads, seq_len, seq_len)

    return causal_mask


def create_sdpa_full_mask(seq_len, device="cpu", num_heads=1):
    """
    创建SDPA全注意力掩码（无因果）

    Args:
        seq_len: 序列长度
        device: 设备
        num_heads: 注意力头数

    Returns:
        full_mask: (1, num_heads, seq_len, seq_len) 格式的全注意力掩码
    """
    # 全1矩阵
    full_mask = torch.ones(seq_len, seq_len, device=device)

    # 转换为SDPA格式：0表示可见
    full_mask = torch.zeros_like(full_mask, dtype=torch.float)

    # 扩展为SDPA期望的形状
    full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    if num_heads > 1:
        full_mask = full_mask.expand(1, num_heads, seq_len, seq_len)

    return full_mask


def convert_flex_mask_to_sdpa(flex_mask, seq_len, num_heads=1):
    """
    将flex_attention掩码转换为SDPA格式

    注意：这是一个简化版本，实际转换可能需要更复杂的逻辑
    因为flex_attention使用稀疏掩码表示，而SDPA需要密集掩码

    Args:
        flex_mask: flex_attention的掩码对象
        seq_len: 序列长度
        num_heads: 注意力头数

    Returns:
        sdpa_mask: SDPA格式的掩码 (1, num_heads, seq_len, seq_len)
    """
    # 由于flex_attention的掩码是稀疏表示，这里简化处理
    # 实际使用时需要根据flex_mask的具体结构进行转换

    # 创建默认的因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=flex_mask.device))

    # 转换为SDPA格式
    sdpa_mask = torch.zeros_like(causal_mask, dtype=torch.float).masked_fill_(
        causal_mask == 0, float("-inf")
    )

    # 扩展为SDPA期望的形状
    sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    if num_heads > 1:
        sdpa_mask = sdpa_mask.expand(1, num_heads, seq_len, seq_len)

    return sdpa_mask


def create_sdpa_mask_from_split_info(split_lens, attn_modes, device="cpu", num_heads=1):
    """
    根据分割信息创建SDPA掩码

    Args:
        split_lens: 每个分割的长度列表
        attn_modes: 每个分割的注意力模式列表
        device: 设备
        num_heads: 注意力头数

    Returns:
        sdpa_mask: SDPA格式的掩码
    """
    return create_sdpa_attention_mask(split_lens, attn_modes, device, num_heads)


def prepare_batch_masks(batch_split_lens, batch_attn_modes, device="cpu", num_heads=1):
    """
    为整个batch准备SDPA掩码，返回4D张量列表

    Args:
        batch_split_lens: batch中每个样本的分割长度列表
        batch_attn_modes: batch中每个样本的注意力模式列表
        device: 设备
        num_heads: 注意力头数

    Returns:
        batch_masks: 4D张量列表，每个形状为 (1, num_heads, seq_len, seq_len)
    """
    batch_masks = []
    for split_lens, attn_modes in zip(batch_split_lens, batch_attn_modes):
        mask = create_sdpa_attention_mask(split_lens, attn_modes, device, num_heads)
        batch_masks.append(mask)

    return batch_masks
