import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import PatchReparam
import torch.nn as nn
from omegaconf import OmegaConf
import torch
from peft import LoraConfig, TaskType

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        use_ema = config.get("use_ema", True)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"/"ema"
        ckpt_key = "ema" if use_ema else "model"
        if ckpt_key in state_dict:
            state_dict = state_dict[ckpt_key]
        elif "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path} (key={ckpt_key})')
    return model


def apply_lora_to_encoder(encoder, lora_config_dict):
    """将 LoRA 配置应用到 encoder"""
    from peft import get_peft_model
    if hasattr(encoder, 'peft_config'):
        print("Encoder already has LoRA configuration. Skipping LoRA application.")
        return encoder
    print("Applying LoRA configuration to encoder...")
    # 处理 lora_config_dict
    if isinstance(lora_config_dict, dict):
        # 处理 task_type
        if 'default' in lora_config_dict:
            peft_config = lora_config_dict['default']
            # 如果 default 的值已经是 LoraConfig 对象，直接使用
            if isinstance(peft_config, LoraConfig):
                print(f"Using LoraConfig from 'default' key")
            else:
                # 如果是字典，用它创建 LoraConfig
                lora_config_dict = peft_config
                peft_config = None
        else:
            peft_config = None
        assert peft_config is not None
    else:
        raise ValueError(f"Invalid lora_config_dict type: {type(lora_config_dict)}")
        # 如果还没有创建 peft_config，从字典创建
    encoder = get_peft_model(encoder, peft_config)
    print(f"LoRA config applied: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
          f"target_modules={peft_config.target_modules}")
    return encoder

def instantiate_from_config_with_key(config, model_key: str = "ema", strict: bool = False, skip_encoder: bool = False) -> object:
    """
    Instantiate model from config with support for specifying which model key to use.

    Args:
        config: Configuration dict with "target", "params", and optionally "ckpt"
        model_key: Key to use from checkpoint (e.g., "ema", "model", "state_dict")
        strict: Whether to use strict loading (default: False, allows partial loading)
        skip_encoder: If True, skip loading encoder weights from checkpoint (encoder should be loaded from config)

    Returns:
        Instantiated model with loaded weights
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 如果 checkpoint 使用了 LoRA，先应用 LoRA 配置到 encoder
        if checkpoint["use_lora"] and checkpoint["lora_config"]:
            print("Detected LoRA configuration in checkpoint. Applying LoRA to encoder...")
            model.encoder = apply_lora_to_encoder(model.encoder, checkpoint["lora_config"])

        # Check for multiple model keys and warn
        model_keys = [k for k in checkpoint.keys() if isinstance(checkpoint[k], dict) and any(isinstance(v, torch.Tensor) for v in checkpoint[k].values())]
        if len(model_keys) > 1:
            import warnings
            warnings.warn(f"Checkpoint contains multiple model keys: {model_keys}. Using '{model_key}' as specified.")

        # Extract state dict from specified key
        if model_key in checkpoint:
            state_dict = checkpoint[model_key]
        else:
            # If specified key not found, try common alternatives
            if "ema" in checkpoint:
                import warnings
                warnings.warn(f"Specified key '{model_key}' not found. Falling back to 'ema'.")
                state_dict = checkpoint["ema"]
            elif "model" in checkpoint:
                import warnings
                warnings.warn(f"Specified key '{model_key}' not found. Falling back to 'model'.")
                state_dict = checkpoint["model"]
            else:
                raise KeyError(f"Specified key '{model_key}' not found in checkpoint. Available keys: {list(checkpoint.keys())}")

        # 如果 skip_encoder=True，过滤掉 encoder 相关的权重
        if skip_encoder:
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
            if len(filtered_state_dict) < len(state_dict):
                skipped_count = len(state_dict) - len(filtered_state_dict)
                print(f"Skipping {skipped_count} encoder weights. Loading {len(filtered_state_dict)} weights (decoder and other components only).")
            state_dict = filtered_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        if missing_keys:
            print(len(missing_keys))
            print(f"Warning: Missing keys when loading {ckpt_path}: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading {ckpt_path}: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Warning: Unexpected keys: {unexpected_keys}")
        print(f'target {config["target"]} loaded from {ckpt_path} (key: {model_key}, skip_encoder: {skip_encoder})')
    return model