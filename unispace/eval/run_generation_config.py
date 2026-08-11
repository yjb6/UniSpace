"""Run UniSpace image generation from a public JSON configuration."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _require(path: str, description: str, child: str | None = None) -> None:
    candidate = Path(path) / child if child else Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Missing {description}: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and print the command without loading the model.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--use-test-prompts", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = _expand(json.load(handle))

    model_path = config["ckpt_path"]
    llm_path = config["llm_path"]
    vae_path = config["vae_path"]
    output_dir = os.path.join(
        config.get("output_base_dir", "./eval/results"),
        config.get("step_name", Path(model_path).name),
        "smoke",
    )

    _require(model_path, "UniSpace checkpoint", "model.safetensors")
    _require(llm_path, "Qwen3 base model")
    _require(vae_path, "Qwen3Unified VAE config")

    if not args.check_only and not (args.prompt or args.use_test_prompts):
        parser.error("one of --prompt or --use-test-prompts is required")

    command = [
        sys.executable,
        "-m",
        "eval.gen.gen_images_qwen3_unified_mot",
        "--output_dir",
        output_dir,
        "--model-path",
        model_path,
        "--llm-path",
        llm_path,
        "--vae-path",
        vae_path,
        "--resolution",
        str(config.get("resolution", 1024)),
        "--max_latent_size",
        str(config.get("max_latent_size", 96)),
        "--num_images_per_prompt",
        "1",
        "--batch_size",
        "1",
        "--cfg_scale",
        str(config.get("cfg_scale", 10)),
        "--inference-steps",
        str(config.get("inference_steps", 50)),
    ]

    for key, value in config.get("extra_gen_args", {}).items():
        command.append(key)
        if value != "":
            command.append(str(value))

    if args.prompt:
        command.extend(["--prompt", args.prompt])
    elif args.use_test_prompts:
        command.append("--use_meigen_test_cases")

    print("Running:", " ".join(command), flush=True)
    if args.check_only:
        print("Preflight passed.", flush=True)
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
