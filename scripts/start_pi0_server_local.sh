#!/usr/bin/env bash
# Start the local OpenPI policy server in a tmux session.
# Defaults to GPU 1 so GPU 0 remains available for vLLM.
#
# Usage:
#   bash scripts/start_pi0_server_local.sh
#   bash scripts/start_pi0_server_local.sh --list
#   bash scripts/start_pi0_server_local.sh pi05_turn_on_tap checkpoints/pi05_turn_on_tap/pi05_turn_on_tap_20260317_180242/49999

set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/config/servers.toml"

_cfg() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1"; }
_cfg_optional() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1" 2>/dev/null || true; }

DEFAULT_OPENPI_ROOT="$REPO_ROOT/$(_cfg pi0.local.openpi_root)"
DEFAULT_POLICY_CONFIG="$(_cfg_optional pi0.local.policy_config)"
if [[ -z "$DEFAULT_POLICY_CONFIG" ]]; then
  DEFAULT_POLICY_CONFIG="$(_cfg pi0.policy_config)"
fi

SESSION_NAME="${SESSION_NAME:-$(_cfg pi0.local.session_name)}"
OPENPI_ROOT="${OPENPI_ROOT:-$DEFAULT_OPENPI_ROOT}"
POLICY_CONFIG="${POLICY_CONFIG:-$DEFAULT_POLICY_CONFIG}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(_cfg pi0.local.checkpoint)}"
PORT="${PORT:-$(_cfg pi0.port)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(_cfg pi0.local.cuda_visible_devices)}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
RECORD="${RECORD:-0}"
PROGRESS_SOURCE="${PROGRESS_SOURCE:-task}"
UV_CMD="${UV_CMD:-uv}"

case "$PROGRESS_SOURCE" in
  task|subtask)
    ;;
  *)
    echo "PROGRESS_SOURCE must be one of: task, subtask." >&2
    exit 1
    ;;
esac

print_usage() {
  cat <<EOF
Usage:
  bash scripts/start_pi0_server_local.sh
  bash scripts/start_pi0_server_local.sh --list
  bash scripts/start_pi0_server_local.sh <policy_config> <checkpoint_dir>

Examples:
  bash scripts/start_pi0_server_local.sh
  bash scripts/start_pi0_server_local.sh --list
  bash scripts/start_pi0_server_local.sh pi05_turn_on_tap checkpoints/pi05_turn_on_tap/pi05_turn_on_tap_20260317_180242/49999

Environment overrides:
  SESSION_NAME         tmux session name (default: $SESSION_NAME)
  OPENPI_ROOT          OpenPI repo root (default: $OPENPI_ROOT)
  POLICY_CONFIG        OpenPI config name (default: $POLICY_CONFIG)
  CHECKPOINT_DIR       Checkpoint path, relative to OPENPI_ROOT or absolute (default: $CHECKPOINT_DIR)
  PORT                 WebSocket port (default: $PORT)
  CUDA_VISIBLE_DEVICES GPU selection for OpenPI (default: $CUDA_VISIBLE_DEVICES)
  DEFAULT_PROMPT       Optional fallback prompt passed to serve_policy.py
  RECORD               Set to 1 to enable policy recording
  PROGRESS_SOURCE      Progress head exposed as action["progress"]: task|subtask (default: $PROGRESS_SOURCE)
  UV_CMD               uv executable or absolute path (default: $UV_CMD)
EOF
}

validate_openpi_root() {
  if [[ ! -d "$OPENPI_ROOT" ]]; then
    echo "OpenPI root '$OPENPI_ROOT' does not exist." >&2
    exit 1
  fi

  if [[ ! -f "$OPENPI_ROOT/scripts/serve_policy.py" ]]; then
    echo "Could not find '$OPENPI_ROOT/scripts/serve_policy.py'." >&2
    exit 1
  fi
}

resolve_checkpoint_dir() {
  local checkpoint_dir="$1"

  if [[ "$checkpoint_dir" != /* ]]; then
    checkpoint_dir="$OPENPI_ROOT/$checkpoint_dir"
  fi

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Checkpoint directory '$checkpoint_dir' does not exist." >&2
    echo "Run 'bash scripts/start_pi0_server_local.sh --list' to inspect available checkpoints." >&2
    exit 1
  fi

  if [[ ! -d "$checkpoint_dir/params" ]]; then
    echo "Checkpoint directory '$checkpoint_dir' does not contain a 'params' subdirectory." >&2
    exit 1
  fi

  realpath "$checkpoint_dir"
}

validate_checkpoint_assets() {
  local checkpoint_path="$1"
  local assets_dir="$checkpoint_path/assets"
  local norm_stats_path="$assets_dir/$POLICY_CONFIG/norm_stats.json"

  [[ -d "$assets_dir" ]] || return 0
  [[ -f "$norm_stats_path" ]] && return 0

  local available_asset_ids=""
  while IFS= read -r asset_dir; do
    local asset_id
    asset_id="$(basename "$asset_dir")"
    if [[ -z "$available_asset_ids" ]]; then
      available_asset_ids="$asset_id"
    else
      available_asset_ids="$available_asset_ids, $asset_id"
    fi
  done < <(find "$assets_dir" -mindepth 1 -maxdepth 1 -type d | sort)

  echo "Checkpoint '$checkpoint_path' is missing norm stats for policy config '$POLICY_CONFIG'." >&2
  if [[ -n "$available_asset_ids" ]]; then
    echo "Available asset IDs under '$assets_dir': $available_asset_ids" >&2
  else
    echo "No asset directories were found under '$assets_dir'." >&2
  fi
  echo "Set POLICY_CONFIG to a matching config name or update config/servers.toml." >&2
  exit 1
}

list_checkpoints() {
  validate_openpi_root

  local checkpoint_root="$OPENPI_ROOT/checkpoints"
  if [[ ! -d "$checkpoint_root" ]]; then
    echo "Checkpoint root '$checkpoint_root' does not exist." >&2
    exit 1
  fi

  local found=0
  while IFS= read -r checkpoint_dir; do
    found=1
    local relative_dir="${checkpoint_dir#$OPENPI_ROOT/}"
    local config_name="${relative_dir#checkpoints/}"
    config_name="${config_name%%/*}"
    printf '%-40s %s\n' "$config_name" "$relative_dir"
  done < <(
    find "$checkpoint_root" -mindepth 3 -maxdepth 3 -type d | sort | while IFS= read -r dir; do
      [[ -d "$dir/params" ]] || continue
      printf '%s\n' "$dir"
    done
  )

  if [[ "$found" -eq 0 ]]; then
    echo "No runnable checkpoints were found under '$checkpoint_root'." >&2
    exit 1
  fi
}

check_uv_command() {
  if [[ "$UV_CMD" == */* ]]; then
    if [[ ! -x "$UV_CMD" ]]; then
      echo "uv binary not found at '$UV_CMD'." >&2
      exit 1
    fi
    return
  fi

  if ! command -v "$UV_CMD" >/dev/null 2>&1; then
    echo "uv command '$UV_CMD' is not installed or not on PATH." >&2
    exit 1
  fi
}

run_server() {
  validate_openpi_root
  check_uv_command

  local checkpoint_path
  checkpoint_path="$(resolve_checkpoint_dir "$CHECKPOINT_DIR")"
  validate_checkpoint_assets "$checkpoint_path"

  export CUDA_VISIBLE_DEVICES

  cd "$OPENPI_ROOT"

  local cmd=(
    "$UV_CMD" run scripts/serve_policy.py
    "--port=$PORT"
  )

  if [[ -n "$DEFAULT_PROMPT" ]]; then
    cmd+=("--default-prompt=$DEFAULT_PROMPT")
  fi

  if [[ "$RECORD" == "1" ]]; then
    cmd+=("--record")
  fi

  cmd+=(
    policy:checkpoint
    "--policy.config=$POLICY_CONFIG"
    "--policy.dir=$checkpoint_path"
    "--policy.progress-source=$PROGRESS_SOURCE"
  )

  echo "OpenPI root: $OPENPI_ROOT"
  echo "Policy config: $POLICY_CONFIG"
  echo "Checkpoint: $checkpoint_path"
  echo "Progress source: $PROGRESS_SOURCE"
  echo "Port: $PORT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

  "${cmd[@]}"
}

INSIDE_TMUX=0
if [[ "${1:-}" == "__run_inside_tmux" ]]; then
  INSIDE_TMUX=1
  shift
fi

case "${1:-}" in
  "")
    ;;
  --list)
    if [[ "$#" -ne 1 ]]; then
      print_usage >&2
      exit 1
    fi
    list_checkpoints
    exit 0
    ;;
  -h|--help)
    print_usage
    exit 0
    ;;
  *)
    if [[ "$#" -ne 2 ]]; then
      print_usage >&2
      exit 1
    fi
    POLICY_CONFIG="$1"
    CHECKPOINT_DIR="$2"
    ;;
esac

if [[ "$INSIDE_TMUX" -eq 1 ]]; then
  run_server
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

validate_openpi_root

ENV_PREFIX=""
for var_name in \
  PATH \
  PYTHONPATH \
  LD_LIBRARY_PATH \
  CONDA_PREFIX \
  CONDA_DEFAULT_ENV \
  VIRTUAL_ENV \
  HF_HOME \
  TRANSFORMERS_CACHE \
  OPENPI_DATA_HOME \
  XLA_FLAGS \
  XLA_PYTHON_CLIENT_MEM_FRACTION \
  CUDA_VISIBLE_DEVICES \
  OPENPI_ROOT \
  POLICY_CONFIG \
  CHECKPOINT_DIR \
  PORT \
  DEFAULT_PROMPT \
  RECORD \
  PROGRESS_SOURCE \
  UV_CMD; do
  if [[ -v "$var_name" ]]; then
    printf -v ENV_PREFIX '%s%s=%q ' "$ENV_PREFIX" "$var_name" "${!var_name}"
  fi
done

printf -v INNER_COMMAND '%s%q __run_inside_tmux' "$ENV_PREFIX" "$SCRIPT_PATH"
printf -v TMUX_COMMAND 'bash -lc %q' "$INNER_COMMAND; exec bash"

echo "Stopping existing '$SESSION_NAME' session (if any)..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "Starting local OpenPI server in tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "$TMUX_COMMAND"

echo "Server starting in tmux session '$SESSION_NAME'."
echo "Policy config: $POLICY_CONFIG"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Progress source: $PROGRESS_SOURCE"
echo "Port: $PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "To view logs: tmux attach -t $SESSION_NAME"
