# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from PIL import Image

# import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from .data_utils import MULTI_RESOLUTION_MAP

class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        # # 修改为固定大小的 resize
        # self.resize_transform = transforms.Resize(
        #     size=(max_image_size, max_image_size),
        #     interpolation=InterpolationMode.BICUBIC,
        #     antialias=True
        # )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        # img = self.resize_transform(img, img_num=img_num)
        img = self.resize_transform(img)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


class MultiResolutionImageTransform:
    """
    多分辨率Transform - 通过微调单边精确匹配目标像素数

    策略:
    1. 按原始纵横比计算初始分辨率(stride对齐)
    2. 如果像素数偏差较大,微调较短边来补偿
    3. 允许纵横比有轻微偏差(通常<5%)

    Args:
        target_pixels: 目标像素总数 (如 512*512 = 262144)
        stride: 分辨率必须是stride的倍数
        max_aspect_deviation: 允许的最大纵横比偏差比例 (默认0.05 = 5%)
        prefer_adjust_short_edge: 优先调整短边 (True) 还是长边 (False)
    """
    def __init__(
        self,
        target_pixels=512*512,
        stride=16,
        max_aspect_deviation=0.05,  # 允许5%的纵横比偏差
        prefer_adjust_short_edge=True,  # 优先调整短边
        aspect_ratio_buckets=None,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ):
        self.target_pixels = target_pixels
        self.stride = stride
        self.max_aspect_deviation = max_aspect_deviation
        self.prefer_adjust_short_edge = prefer_adjust_short_edge
        self.interpolation = interpolation
        self.antialias = antialias

        # 处理 aspect_ratio_buckets 参数
        if aspect_ratio_buckets is not None:
            # 如果是字符串，从 MULTI_RESOLUTION_MAP 索引
            if isinstance(aspect_ratio_buckets, str):
                if aspect_ratio_buckets not in MULTI_RESOLUTION_MAP:
                    raise ValueError(
                        f"Unknown aspect_ratio_buckets key: {aspect_ratio_buckets}. "
                        f"Available keys: {list(MULTI_RESOLUTION_MAP.keys())}"
                    )
                bucket_dict = MULTI_RESOLUTION_MAP[aspect_ratio_buckets]
            else:
                # 如果直接传入字典，直接使用
                bucket_dict = aspect_ratio_buckets

            self.use_buckets = True
            self.aspect_ratios = sorted([float(k) for k in bucket_dict.keys()])
            self.resolutions = {float(k): v for k, v in bucket_dict.items()}
        else:
            self.use_buckets = False

        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean, std=image_std, inplace=True
        )

    def _make_divisible(self, value, stride):
        """确保值是stride的倍数"""
        return max(stride, int(round(value / stride) * stride))

    def _calculate_target_size_dynamic(self, width, height):
        """
        动态计算目标尺寸 - 通过微调单边精确匹配目标像素数

        算法步骤:
        1. 按原始纵横比计算初始分辨率
        2. 计算实际像素数与目标的差距
        3. 固定一边,调整另一边来补偿差距
        4. 确保调整后的纵横比偏差在允许范围内
        """
        original_aspect_ratio = width / height

        # Step 1: 计算初始分辨率(保持纵横比)
        ideal_width = (self.target_pixels * original_aspect_ratio) ** 0.5
        ideal_height = (self.target_pixels / original_aspect_ratio) ** 0.5

        # Stride对齐
        base_width = self._make_divisible(ideal_width, self.stride)
        base_height = self._make_divisible(ideal_height, self.stride)

        # Step 2: 计算像素差距
        base_pixels = base_width * base_height
        pixel_diff = self.target_pixels - base_pixels

        # 如果差距在±5%以内,直接返回
        if abs(pixel_diff) / self.target_pixels < 0.05:
            return int(base_width), int(base_height)

        # Step 3: 微调策略
        is_wide_image = width >= height

        if self.prefer_adjust_short_edge:
            # 优先调整短边(对视觉影响更小)
            if is_wide_image:
                # 宽图 - 调整高度
                adjusted_width, adjusted_height = self._adjust_dimension(
                    fixed_dim=base_width,
                    adjustable_dim=base_height,
                    target_pixels=self.target_pixels,
                    is_width_fixed=True,
                    original_aspect=original_aspect_ratio
                )
            else:
                # 竖图 - 调整宽度
                adjusted_width, adjusted_height = self._adjust_dimension(
                    fixed_dim=base_height,
                    adjustable_dim=base_width,
                    target_pixels=self.target_pixels,
                    is_width_fixed=False,
                    original_aspect=original_aspect_ratio
                )
        else:
            # 优先调整长边(可能获得更好的像素匹配)
            if is_wide_image:
                # 宽图 - 调整宽度
                adjusted_width, adjusted_height = self._adjust_dimension(
                    fixed_dim=base_height,
                    adjustable_dim=base_width,
                    target_pixels=self.target_pixels,
                    is_width_fixed=False,
                    original_aspect=original_aspect_ratio
                )
            else:
                # 竖图 - 调整高度
                adjusted_width, adjusted_height = self._adjust_dimension(
                    fixed_dim=base_width,
                    adjustable_dim=base_height,
                    target_pixels=self.target_pixels,
                    is_width_fixed=True,
                    original_aspect=original_aspect_ratio
                )

        # Step 4: 验证纵横比偏差
        adjusted_aspect = adjusted_width / adjusted_height
        aspect_deviation = abs(adjusted_aspect - original_aspect_ratio) / original_aspect_ratio

        if aspect_deviation > self.max_aspect_deviation:
            # 如果纵横比偏差太大,回退到基础方案
            return int(base_width), int(base_height)

        return int(adjusted_width), int(adjusted_height)

    def _adjust_dimension(self, fixed_dim, adjustable_dim, target_pixels,
                         is_width_fixed, original_aspect):
        """
        固定一个维度,调整另一个维度来匹配目标像素数

        Args:
            fixed_dim: 固定的维度值
            adjustable_dim: 要调整的维度值
            target_pixels: 目标像素数
            is_width_fixed: 固定的是宽度(True)还是高度(False)
            original_aspect: 原始纵横比

        Returns:
            (adjusted_width, adjusted_height)
        """
        # 计算理论上需要的另一维度值
        required_dim = target_pixels / fixed_dim

        # Stride对齐
        aligned_dim = self._make_divisible(required_dim, self.stride)

        # 如果对齐后偏差太大,尝试相邻的stride值
        actual_pixels_1 = fixed_dim * aligned_dim
        deviation_1 = abs(actual_pixels_1 - target_pixels) / target_pixels

        # 尝试 ±1 stride
        alternative_dims = [
            aligned_dim - self.stride,
            aligned_dim + self.stride
        ]

        best_dim = aligned_dim
        best_deviation = deviation_1

        for alt_dim in alternative_dims:
            if alt_dim < self.stride:
                continue

            alt_pixels = fixed_dim * alt_dim
            alt_deviation = abs(alt_pixels - target_pixels) / target_pixels

            if alt_deviation < best_deviation:
                best_deviation = alt_deviation
                best_dim = alt_dim

        # 返回调整后的宽高
        if is_width_fixed:
            return fixed_dim, best_dim
        else:
            return best_dim, fixed_dim

    def _calculate_target_size_bucket(self, width, height):
        """使用预定义桶计算目标尺寸"""
        aspect_ratio = width / height
        closest_ratio = min(
            self.aspect_ratios,
            key=lambda x: abs(x - aspect_ratio)
        )
        target_resolution = self.resolutions[closest_ratio]
        return int(target_resolution[0]), int(target_resolution[1])

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        if self.use_buckets:
            target_width, target_height = self._calculate_target_size_bucket(width, height)
        else:
            target_width, target_height = self._calculate_target_size_dynamic(width, height)

        img = F.resize(
            img,
            (target_height, target_width),
            self.interpolation,
            antialias=self.antialias
        )

        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)

        return img

    def get_target_size(self, width, height):
        """获取目标尺寸(不执行实际转换)"""
        if self.use_buckets:
            return self._calculate_target_size_bucket(width, height)
        else:
            return self._calculate_target_size_dynamic(width, height)

    def get_adjustment_info(self, width, height):
        """
        获取调整信息(用于调试和分析)

        Returns:
            dict: {
                'original_size': (width, height),
                'original_aspect': float,
                'target_size': (w, h),
                'target_aspect': float,
                'aspect_deviation': float,
                'pixel_deviation': float,
            }
        """
        target_w, target_h = self.get_target_size(width, height)
        original_aspect = width / height
        target_aspect = target_w / target_h

        return {
            'original_size': (width, height),
            'original_aspect': original_aspect,
            'original_pixels': width * height,
            'target_size': (target_w, target_h),
            'target_aspect': target_aspect,
            'target_pixels': target_w * target_h,
            'aspect_deviation': abs(target_aspect - original_aspect) / original_aspect,
            'pixel_deviation': abs(target_w * target_h - self.target_pixels) / self.target_pixels,
        }


def decolorization(image):
    gray_image = image.convert('L')
    return Image.merge(image.mode, [gray_image] * 3) if image.mode in ('RGB', 'L') else gray_image


def downscale(image, scale_factor):
    new_width = int(round(image.width * scale_factor))
    new_height = int(round(image.height * scale_factor))
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def crop(image, crop_factors):
    target_h, target_w = crop_factors
    img_w, img_h = image.size

    if target_h > img_h or target_w > img_w:
        raise ValueError("Crop size exceeds image dimensions")

    x = random.randint(0, img_w - target_w)
    y = random.randint(0, img_h - target_h)

    return image.crop((x, y, x + target_w, y + target_h)), [[x, y], [x + target_w, y + target_h]]


def motion_blur_opencv(image, kernel_size=15, angle=0):
    # 线性核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)

    # 旋转核
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # 归一化核
    rotated_kernel /= rotated_kernel.sum() if rotated_kernel.sum() != 0 else 1

    img = np.array(image)
    if img.ndim == 2:
        blurred = cv2.filter2D(img, -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)
    else:
        # 对于彩色图像，各通道独立卷积
        blurred = np.zeros_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = cv2.filter2D(img[..., c], -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)

    return Image.fromarray(blurred.astype(np.uint8))


def shuffle_patch(image, num_splits, gap_size=2):
    """将图像分割为块（允许尺寸不整除），随机打乱后拼接，块间保留间隙"""
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    random.shuffle(patches)

    total_width = sum(patch_widths) + (w_splits - 1) * gap_size
    total_height = sum(patch_heights) + (h_splits - 1) * gap_size
    new_image = Image.new(image.mode, (total_width, total_height), color=(255, 255, 255))

    current_y = 0  # 当前行的起始 Y 坐标
    patch_idx = 0  # 当前处理的块索引
    for i in range(h_splits):
        current_x = 0  # 当前列的起始 X 坐标
        patch_h = patch_heights[i]  # 当前行块的高度
        for j in range(w_splits):
            # 取出打乱后的块
            patch = patches[patch_idx]
            patch_w = patch_widths[j]  # 当前列块的宽度
            # 粘贴块（左上角坐标为 (current_x, current_y)）
            new_image.paste(patch, (current_x, current_y))
            # 更新 X 坐标（下一个块的起始位置 = 当前块宽度 + 间隙）
            current_x += patch_w + gap_size
            patch_idx += 1
        # 更新 Y 坐标（下一行的起始位置 = 当前行高度 + 间隙）
        current_y += patch_h + gap_size

    return new_image


def inpainting(image, num_splits, blank_ratio=0.3, blank_color=(255, 255, 255)):
    """
    图像分割后随机空白部分patch，用于inpainting任务

    参数：
        image: PIL.Image 输入图像（RGB模式）
        h_splits: int 行分割数（垂直方向分割块数）
        w_splits: int 列分割数（水平方向分割块数）
        blank_ratio: float 空白patch的比例（0~1）
        blank_color: tuple 空白区域的颜色（RGB，如白色(255,255,255)）

    返回：
        PIL.Image 处理后拼接的图像
    """
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop((current_x, current_y, current_x + patch_w, current_y + patch_h))
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    total_patches = h_splits * w_splits
    num_blank = int(total_patches * blank_ratio)
    num_blank = max(0, min(num_blank, total_patches))
    blank_indices = random.sample(range(total_patches), num_blank)

    processed_patches = []
    for idx, patch in enumerate(patches):
        if idx in blank_indices:
            blank_patch = Image.new("RGB", patch.size, color=blank_color)
            processed_patches.append(blank_patch)
        else:
            processed_patches.append(patch)

    # 创建结果图像（尺寸与原图一致）
    result_image = Image.new("RGB", (img_w, img_h))
    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            # 取出处理后的patch
            patch = processed_patches[patch_idx]
            patch_w = patch_widths[j]
            # 粘贴到原位置
            result_image.paste(patch, (current_x, current_y))
            current_x += patch_w
            patch_idx += 1
        current_y += patch_h

    return result_image
