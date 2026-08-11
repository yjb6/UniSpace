#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: run_gen_gpu_qwen3_mot.sh CONFIG [PROMPT]}
PROMPT=${2:-"A red panda reading a book beside a window."}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$PROJECT_ROOT"
python -m eval.run_generation_config --config "$CONFIG" --prompt "$PROMPT"
