#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/config/servers.toml"

_cfg() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1"; }

MODEL_PATH="$(_cfg vllm.local.model_path)"
SERVED_MODEL_NAME="$(_cfg vllm.local.served_model_name)"

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len 262144 \
    --reasoning-parser qwen3
