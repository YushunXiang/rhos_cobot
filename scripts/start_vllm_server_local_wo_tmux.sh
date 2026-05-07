#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/config/servers.toml"

_cfg() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1"; }
_cfg_optional() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1" 2>/dev/null || true; }

DEFAULT_VLLM_MAX_MODEL_LEN="$(_cfg_optional vllm.local.max_model_len)"
if [[ -z "$DEFAULT_VLLM_MAX_MODEL_LEN" ]]; then
  DEFAULT_VLLM_MAX_MODEL_LEN="262144"
fi

DEFAULT_VLLM_MAX_NUM_SEQS="$(_cfg_optional vllm.local.max_num_seqs)"
if [[ -z "$DEFAULT_VLLM_MAX_NUM_SEQS" ]]; then
  DEFAULT_VLLM_MAX_NUM_SEQS="512"
fi

DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="$(_cfg_optional vllm.local.gpu_memory_utilization)"

MODEL_PATH="$(_cfg vllm.local.model_path)"
SERVED_MODEL_NAME="$(_cfg vllm.local.served_model_name)"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$DEFAULT_VLLM_MAX_MODEL_LEN}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$DEFAULT_VLLM_MAX_NUM_SEQS}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"

validate_positive_int() {
  local name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "$name must be a positive integer." >&2
    exit 2
  fi
}

validate_positive_int "VLLM_MAX_MODEL_LEN" "$VLLM_MAX_MODEL_LEN"
validate_positive_int "VLLM_MAX_NUM_SEQS" "$VLLM_MAX_NUM_SEQS"

serve_args=(
    serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --reasoning-parser qwen3
)

if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    serve_args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

vllm "${serve_args[@]}"
