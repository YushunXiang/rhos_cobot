#!/usr/bin/env bash
# Start the local vLLM server in a tmux session.
# Usage: bash scripts/start_vllm_server_local.sh

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

SESSION_NAME="${SESSION_NAME:-$(_cfg vllm.local.session_name)}"
MODEL_PATH="${MODEL_PATH:-$(_cfg vllm.local.model_path)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(_cfg vllm.local.served_model_name)}"
HOST="${HOST:-$(_cfg vllm.host)}"
PORT="${PORT:-$(_cfg vllm.port)}"
VLLM_CMD="${VLLM_CMD:-vllm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-$(_cfg vllm.local.conda_env_name)}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
CONDA_SH="${CONDA_SH:-${CONDA_BASE:+$CONDA_BASE/etc/profile.d/conda.sh}}"
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

run_server() {
  local -a serve_args

  validate_positive_int "VLLM_MAX_MODEL_LEN" "$VLLM_MAX_MODEL_LEN"
  validate_positive_int "VLLM_MAX_NUM_SEQS" "$VLLM_MAX_NUM_SEQS"

  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"

  if [[ "$VLLM_CMD" == */* ]]; then
    if [[ ! -x "$VLLM_CMD" ]]; then
      echo "vLLM binary not found at '$VLLM_CMD'." >&2
      exit 1
    fi
  elif ! command -v "$VLLM_CMD" >/dev/null 2>&1; then
    echo "vLLM command '$VLLM_CMD' is not installed or not on PATH after activating '$CONDA_ENV_NAME'." >&2
    exit 1
  fi

  serve_args=(
    serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --reasoning-parser qwen3
    --host "$HOST"
    --port "$PORT"
  )
  if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    serve_args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
  fi

  "$VLLM_CMD" "${serve_args[@]}"
}

if [[ "${1:-}" == "__run_inside_tmux" ]]; then
  run_server
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

if [[ -z "$CONDA_BASE" ]]; then
  echo "Could not determine the Conda base path. Ensure 'conda' is installed and on PATH." >&2
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found at '$CONDA_SH'." >&2
  exit 1
fi

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
ENV_PREFIX=""
for var_name in \
  PATH \
  PYTHONPATH \
  LD_LIBRARY_PATH \
  CONDA_PREFIX \
  CONDA_DEFAULT_ENV \
  VIRTUAL_ENV \
  CUDA_VISIBLE_DEVICES \
  HF_HOME \
  TRANSFORMERS_CACHE \
  MODEL_PATH \
  SERVED_MODEL_NAME \
  HOST \
  PORT \
  VLLM_MAX_MODEL_LEN \
  VLLM_MAX_NUM_SEQS \
  VLLM_GPU_MEMORY_UTILIZATION \
  CONDA_ENV_NAME \
  CONDA_BASE \
  CONDA_SH \
  VLLM_CMD; do
  if [[ -v "$var_name" ]]; then
    printf -v ENV_PREFIX '%s%s=%q ' "$ENV_PREFIX" "$var_name" "${!var_name}"
  fi
done

printf -v INNER_COMMAND '%s%q __run_inside_tmux' "$ENV_PREFIX" "$SCRIPT_PATH"
printf -v TMUX_COMMAND 'bash -lc %q' "$INNER_COMMAND; exec bash"

echo "Stopping existing '$SESSION_NAME' session (if any)..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "Starting local vLLM server in tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "$TMUX_COMMAND"

echo "Server starting in tmux session '$SESSION_NAME'."
echo "Expected base URL: http://$HOST:$PORT/v1"
echo "Expected model: $SERVED_MODEL_NAME"
echo "To view logs: tmux attach -t $SESSION_NAME"
