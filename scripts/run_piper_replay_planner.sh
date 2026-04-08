#!/usr/bin/env bash
# Run examples.piper_real.main in offline HDF5 replay + VLM planner mode.
#
# Usage:
#   bash scripts/run_piper_replay_planner.sh
#   bash scripts/run_piper_replay_planner.sh local
#   PLANNER_HOST=192.168.3.123 bash scripts/run_piper_replay_planner.sh remote
#   START_SERVERS=0 bash scripts/run_piper_replay_planner.sh none -- --planner.model Qwen/Qwen3.5-4B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/run_piper_replay_planner.sh [wrapper options] [local|remote|none] [--] [extra main.py args...]

Modes:
  local   Start the planner server locally (default)
  remote  Start the planner server remotely
  none    Do not start servers; connect to an already running planner server

Wrapper options:
  --kill-existing-replay
                     Stop any existing replay planner process before starting
                     default: enabled
  --no-kill-existing-replay
                     Keep any existing replay planner process alive
  --replay-kill-grace-sec SECONDS
                     Wait SECONDS after SIGTERM before SIGKILL
                     default: 5
  -h, --help         Show this help message

Forwarded args:
  Unknown options and args are forwarded to examples.piper_real.main.
  Use '--' to force all remaining args to be forwarded unchanged.

Environment overrides:
  DATASET            Replay dataset path
                     default: /inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/aloha-data/long-horizon-demo/episode_4.hdf5
  PROMPT             Prompt forwarded to examples.piper_real.main
                     default: long-horizon replay planner validation
  TASK_NAME          Resolve PROMPT from config/task_prompts.json when PROMPT is unset
  MAX_EPISODE_STEPS  Replay frame limit forwarded to --max-episode-steps
                     default: 0 (use full dataset)
  PLANNER_HOST       Planner server host
                     default: 127.0.0.1 for local/none; required for remote
  PLANNER_PORT       Planner server port
                     default: value from config/servers.toml
  PLANNER_MODEL      Planner model name
                     default: value from config/servers.toml for the selected mode
  OPENPI_ROOT        Local OpenPI repo root used to resolve openpi_client
                     default: value from config/servers.toml -> pi0.local.openpi_root
  PYTHON_CMD         Python interpreter used to run main.py
                     default: examples/piper_real/.venv/bin/python, fallback: python3
  START_SERVERS      Set to 0 to skip server startup even in local/remote mode
                     default: 1
  START_TARGET       Which startup helper to use in local/remote mode
                     values: planner, all
                     default: planner
  NAVIGATION_ONLY    Set to 0 to keep manipulate subtasks in the decomposition log
                     default: 1
  WAIT_FOR_PLANNER_READY
                     Set to 0 to skip the preflight wait loop
                     default: 1
  PLANNER_READY_TIMEOUT_SEC
                     Max seconds to wait for planner readiness
                     default: 180
  PLANNER_READY_RETRY_INTERVAL_SEC
                     Seconds between readiness probes
                     default: 2
  PLANNER_READY_CHECK_TIMEOUT_SEC
                     Timeout for each individual readiness probe
                     default: 5

Examples:
  bash scripts/run_piper_replay_planner.sh
  bash scripts/run_piper_replay_planner.sh --replay-kill-grace-sec 10
  START_TARGET=all bash scripts/run_piper_replay_planner.sh local
  PLANNER_HOST=192.168.3.123 bash scripts/run_piper_replay_planner.sh remote
  START_SERVERS=0 bash scripts/run_piper_replay_planner.sh none -- --skip-server-checks
EOF
}

MODE="local"
MODE_SET=0
KILL_EXISTING_REPLAY="${KILL_EXISTING_REPLAY:-1}"
REPLAY_KILL_GRACE_SEC="${REPLAY_KILL_GRACE_SEC:-5}"
MAIN_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    local|remote|none)
      if [[ "$MODE_SET" == "1" ]]; then
        echo "Mode was specified more than once." >&2
        exit 2
      fi
      MODE="$1"
      MODE_SET=1
      shift
      ;;
    --kill-existing-replay)
      KILL_EXISTING_REPLAY="1"
      shift
      ;;
    --no-kill-existing-replay)
      KILL_EXISTING_REPLAY="0"
      shift
      ;;
    --replay-kill-grace-sec)
      if [[ "$#" -lt 2 ]]; then
        echo "--replay-kill-grace-sec requires a value." >&2
        exit 2
      fi
      REPLAY_KILL_GRACE_SEC="$2"
      shift 2
      ;;
    --replay-kill-grace-sec=*)
      REPLAY_KILL_GRACE_SEC="${1#*=}"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      MAIN_ARGS+=("$@")
      break
      ;;
    *)
      MAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$KILL_EXISTING_REPLAY" in
  0|1)
    ;;
  *)
    echo "KILL_EXISTING_REPLAY must be 0 or 1." >&2
    exit 2
    ;;
esac

if ! [[ "$REPLAY_KILL_GRACE_SEC" =~ ^[0-9]+$ ]]; then
  echo "REPLAY_KILL_GRACE_SEC must be a non-negative integer." >&2
  exit 2
fi

stop_existing_replay_processes() {
  if [[ "${KILL_EXISTING_REPLAY:-1}" != "1" ]]; then
    return 0
  fi

  local grace_sec="${REPLAY_KILL_GRACE_SEC:-5}"
  local -a replay_pids=()
  local -a remaining_pids=()
  local line pid proc_name

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    pid="${line%% *}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    proc_name="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d '[:space:]')"
    [[ "$proc_name" == python* ]] || continue
    replay_pids+=("$pid")
    echo "Found existing replay planner process: $line"
  done < <(pgrep -af '\-m examples\.piper_real\.main .*--replay-dataset .*--use-llm-planner' || true)

  if [[ "${#replay_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "Stopping existing replay planner process(es): ${replay_pids[*]}"
  kill "${replay_pids[@]}" 2>/dev/null || true

  if (( grace_sec > 0 )); then
    local deadline=$((SECONDS + grace_sec))
    while (( SECONDS < deadline )); do
      remaining_pids=()
      for pid in "${replay_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          remaining_pids+=("$pid")
        fi
      done
      if [[ "${#remaining_pids[@]}" -eq 0 ]]; then
        echo "Existing replay planner process(es) stopped."
        return 0
      fi
      sleep 1
    done
  fi

  remaining_pids=()
  for pid in "${replay_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      remaining_pids+=("$pid")
    fi
  done

  if [[ "${#remaining_pids[@]}" -gt 0 ]]; then
    echo "Force killing replay planner process(es): ${remaining_pids[*]}"
    kill -KILL "${remaining_pids[@]}" 2>/dev/null || true
  fi
}

wait_for_planner_ready() {
  local base_url="$1"
  local model="$2"

  if [[ "${WAIT_FOR_PLANNER_READY:-1}" != "1" ]]; then
    return 0
  fi

  local timeout_sec="${PLANNER_READY_TIMEOUT_SEC:-180}"
  local retry_interval_sec="${PLANNER_READY_RETRY_INTERVAL_SEC:-2}"
  local check_timeout_sec="${PLANNER_READY_CHECK_TIMEOUT_SEC:-5}"
  local deadline=$((SECONDS + timeout_sec))
  local attempt=1
  local check_mode="local"

  if [[ "$MODE" == "remote" ]]; then
    check_mode="remote"
  fi

  echo "Waiting for planner server at $base_url ..."
  while true; do
    if "$PYTHON_CMD" -m examples.piper_real.server_checks \
      --planner-base-url "$base_url" \
      --planner-model "$model" \
      --timeout-sec "$check_timeout_sec" \
      >/dev/null 2>&1; then
      echo "Planner server is ready: $base_url"
      return 0
    fi

    if (( SECONDS >= deadline )); then
      echo "Timed out after ${timeout_sec}s waiting for planner server: $base_url" >&2
      echo "Check status with: bash scripts/check_servers.sh $check_mode" >&2
      return 1
    fi

    echo "Planner not ready yet; retrying in ${retry_interval_sec}s (attempt ${attempt})..."
    attempt=$((attempt + 1))
    sleep "$retry_interval_sec"
  done
}

DATASET="${DATASET:-/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/aloha-data/long-horizon-demo/episode_4.hdf5}"
TASK_NAME="${TASK_NAME:-}"
PROMPT_SOURCE="${PROMPT_SOURCE:-default}"
if [[ -n "${PROMPT:-}" ]]; then
  PROMPT="$PROMPT"
  PROMPT_SOURCE="env:PROMPT"
elif [[ -n "$TASK_NAME" ]]; then
  PROMPT="$(server_lookup_task_prompt "$TASK_NAME")"
  PROMPT_SOURCE="task_catalog:$TASK_NAME"
else
  PROMPT="long-horizon replay planner validation"
fi
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
PYTHON_CMD="${PYTHON_CMD:-$(server_default_python_cmd)}"
START_SERVERS="${START_SERVERS:-1}"
START_TARGET="${START_TARGET:-planner}"
NAVIGATION_ONLY="${NAVIGATION_ONLY:-1}"

server_require_config

OPENPI_ROOT="${OPENPI_ROOT:-$(server_default_openpi_root)}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-}"
PYTHON_CMD="$(server_resolve_python_cmd "$PYTHON_CMD")"
OPENPI_CLIENT_SRC="$(server_resolve_openpi_client_src "$OPENPI_ROOT" "$OPENPI_CLIENT_SRC")"
server_export_openpi_pythonpath "$OPENPI_CLIENT_SRC"

PLANNER_PORT="${PLANNER_PORT:-$(server_cfg vllm.port)}"
if [[ -z "${PLANNER_HOST:-}" ]]; then
  if [[ "$MODE" == "remote" ]]; then
    echo "PLANNER_HOST is required in remote mode. Use the robot workstation reachable address, not the SSH alias." >&2
    exit 1
  fi
  PLANNER_HOST="127.0.0.1"
fi

if [[ -z "${PLANNER_MODEL:-}" ]]; then
  if [[ "$MODE" == "remote" ]]; then
    PLANNER_MODEL="$(server_cfg vllm.remote.served_model_name)"
  else
    PLANNER_MODEL="$(server_cfg vllm.local.served_model_name)"
  fi
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Replay dataset does not exist: $DATASET" >&2
  exit 1
fi

PLANNER_BASE_URL="http://$PLANNER_HOST:$PLANNER_PORT/v1"

cd "$SERVER_REPO_ROOT"
stop_existing_replay_processes

if [[ "$START_SERVERS" == "1" && "$MODE" != "none" ]]; then
  case "$START_TARGET" in
    planner)
      if [[ "$MODE" == "local" ]]; then
        bash "$SCRIPT_DIR/start_vllm_server_local.sh"
      else
        bash "$SCRIPT_DIR/start_vllm_server.sh"
      fi
      ;;
    all)
      bash "$SCRIPT_DIR/start_servers.sh" "$MODE"
      ;;
    *)
      echo "Unsupported START_TARGET='$START_TARGET'. Expected 'planner' or 'all'." >&2
      exit 1
      ;;
  esac
else
  echo "Skipping server startup."
fi

wait_for_planner_ready "$PLANNER_BASE_URL" "$PLANNER_MODEL"

cmd=(
  "$PYTHON_CMD" -m examples.piper_real.main
  --prompt "$PROMPT"
  --replay-dataset "$DATASET"
  --use-llm-planner
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --planner.base-url "$PLANNER_BASE_URL"
  --planner.model "$PLANNER_MODEL"
)

if [[ "$NAVIGATION_ONLY" == "1" ]]; then
  cmd+=(--navigation-only)
fi

if [[ "${#MAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${MAIN_ARGS[@]}")
fi

echo "Mode: $MODE"
echo "Dataset: $DATASET"
echo "Task name: ${TASK_NAME:-<unset>}"
echo "Prompt: $PROMPT"
echo "Prompt source: $PROMPT_SOURCE"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Python: $PYTHON_CMD"
echo "openpi_client src: $OPENPI_CLIENT_SRC"
echo "Planner base URL: $PLANNER_BASE_URL"
echo "Planner model: $PLANNER_MODEL"
echo "Start target: $START_TARGET"
echo "Navigation only: $NAVIGATION_ONLY"
echo "Running:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
