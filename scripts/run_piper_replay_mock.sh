#!/usr/bin/env bash
# Run examples.piper_real.main in server-backed replay mock mode.
#
# Usage:
#   bash scripts/run_piper_replay_mock.sh
#   bash scripts/run_piper_replay_mock.sh local
#   bash scripts/run_piper_replay_mock.sh remote
#   START_SERVERS=0 PYTHON_CMD=../openpi/.venv/bin/python bash scripts/run_piper_replay_mock.sh none --action-horizon 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/run_piper_replay_mock.sh [wrapper options] [local|remote|none|mock] [--] [extra main.py args...]

Modes:
  local   Start the Pi0 policy server locally (default)
  remote  Start the Pi0 policy server remotely
  none    Do not start servers; connect to an already running pi0 policy server
  mock    Start the built-in mock policy server (examples.piper_real.mock_policy_server)
          in the background, bound to PI0_HOST:PI0_PORT; kill it on exit.
          Useful when the real pi0 server is unavailable.

Wrapper options:
  --planner-replay    Delegate to offline VLM planner replay instead of pi0 policy replay
  --hybrid-replay     Run offline replay with VLM navigation + pi0 manipulation
  --policy-replay     Force pi0 policy replay
  --mock-pi0          Deprecated alias for MODE=mock.
  --kill-existing-replay
                     Stop any existing replay mock process before starting
                     default: enabled
  --visualize         Show camera views and subtask overlay during replay
  --save-path PATH    Save replay visualization to MP4 at PATH
  --no-kill-existing-replay
                     Keep any existing replay mock process alive
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
                     default: long-horizon replay mock validation
  TASK_NAME          Resolve PROMPT from config/task_prompts.json when PROMPT is unset
  REPLAY_MODE        Replay backend
                     values: policy, planner, hybrid
                     default: policy
  MAX_EPISODE_STEPS  --max-episode-steps for replay
                     default: 0
  MANIPULATE_MAX_STEPS
                     Safety cap for replay policy steps per manipulate subtask in hybrid mode
                     default: 64
  MANIPULATE_REPLAN_INTERVAL_STEPS
                     Policy-step interval between VLM prompt replans in hybrid mode
                     default: 16
  POLICY_CONFIG      OpenPI policy config used when starting a local/remote pi0 server
                     default: pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3
  CHECKPOINT_DIR     OpenPI checkpoint used when starting a local/remote pi0 server
                     default: /inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/openpi/checkpoints/pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3/pi05_pick_bread_leaf_progress_dual_20260424_061145/99999
  PROGRESS_SOURCE    Progress head exposed by OpenPI as action["progress"]
                     values: task, subtask
                     default: subtask
  PROGRESS_HEAD_MODE Hybrid progress handling in examples.piper_real.main
                     values: auto, force, off
                     default: auto
  TASK_SPEC          Common alias for REPLAY_TASK_SPEC when REPLAY_TASK_SPEC is unset
  REPLAY_TASK_SPEC   Ordered task-spec JSON passed to hybrid ordered-task memory
                     default: config/episode4_plate_wash_sandwich.task_spec.json
                     when REPLAY_MODE=hybrid and DATASET basename is episode_4.hdf5
                     set to 'none' to disable the default auto-spec
  PI0_HOST           pi0 policy server host (policy/hybrid modes)
                     default: 127.0.0.1 for local/none/mock; required for remote
  PI0_PORT           pi0 policy server port (policy/hybrid modes)
                     default: value from config/servers.toml
  PLANNER_HOST       Planner server host (planner/hybrid modes)
                     default: PI0_HOST when set; else 127.0.0.1 for local/none
  PLANNER_PORT       Planner server port (planner/hybrid modes)
                     default: value from config/servers.toml
  PLANNER_MODEL      Planner model name (planner/hybrid modes)
                     default: value from config/servers.toml for the selected mode
  PLANNER_BACKEND    Planner backend (planner/hybrid modes)
                     values: vllm, qz
                     default: vllm
  QZ_STATE_FILE      qz planner state JSON with seq/api_key
                     default: config/servers.toml -> qz.state_file, else
                     config/vllm_server_state.json
  QZ_ENDPOINT_TEMPLATE
                     qz OpenAI-compatible endpoint template
                     default: config/servers.toml -> qz.endpoint_template, else
                     https://qwen35-9b-{seq}-inf.openapi-qb.sii.edu.cn/v1
  PLANNER_REPLANNER_ENABLE_THINKING
                     Enable Qwen thinking for manipulation replanner requests
                     values: 0, 1, true, false
                     default: config/servers.toml -> planner.manipulation_replanner_enable_thinking
  PLANNER_REPLANNER_MAX_TOKENS
                     Max completion tokens for manipulation replanner requests
                     default: config/servers.toml -> planner.manipulation_replanner_max_tokens
  NAVIGATION_ONLY    Filter out manipulate subtasks from the decomposition log
                     (planner mode only). values: 0, 1; default: 1
  OPENPI_ROOT        Local OpenPI repo root used to resolve openpi_client
                     default: value from config/servers.toml -> pi0.local.openpi_root
  PYTHON_CMD         Python interpreter used to run main.py
                     default: examples/piper_real/.venv/bin/python, fallback: python3
  START_SERVERS      Set to 0 to skip server startup even in local/remote mode
                     default: 1
  START_TARGET       Which startup helper to use in local/remote mode
                     values: pi0, planner, all
                     default: pi0 (policy/hybrid), planner (planner)
  SAVE_PATH          Forwarded to main.py --save-path for MP4 output
                     default: empty (disabled)
  VISUALIZE          Forwarded to main.py --visualize when set to 1/true
                     default: 0
  WAIT_FOR_PI0_READY Set to 0 to skip the preflight wait loop
                     default: 1
  PI0_READY_TIMEOUT_SEC
                     Max seconds to wait for pi0 readiness
                     default: 180
  PI0_READY_RETRY_INTERVAL_SEC
                     Seconds between readiness probes
                     default: 2
  PI0_READY_CHECK_TIMEOUT_SEC
                     Timeout for each individual readiness probe
                     default: 5
  WAIT_FOR_PLANNER_READY
                     Set to 0 to skip the planner preflight wait loop in hybrid mode
                     default: 1
  PLANNER_READY_TIMEOUT_SEC
                     Max seconds to wait for planner readiness in hybrid mode
                     default: 180
  PLANNER_READY_RETRY_INTERVAL_SEC
                     Seconds between planner readiness probes in hybrid mode
                     default: 2
  PLANNER_READY_CHECK_TIMEOUT_SEC
                     Timeout for each individual planner readiness probe in hybrid mode
                     default: 5

Examples:
  bash scripts/run_piper_replay_mock.sh
  bash scripts/run_piper_replay_mock.sh mock
  bash scripts/run_piper_replay_mock.sh --replay-kill-grace-sec 10
  bash scripts/run_piper_replay_mock.sh --no-kill-existing-replay none
  REPLAY_MODE=planner START_TARGET=all bash scripts/run_piper_replay_mock.sh
  REPLAY_MODE=hybrid START_TARGET=all bash scripts/run_piper_replay_mock.sh
  bash scripts/run_piper_replay_mock.sh remote
  START_TARGET=all bash scripts/run_piper_replay_mock.sh local
  START_SERVERS=0 PI0_HOST=127.0.0.1 PYTHON_CMD=../openpi/.venv/bin/python \\
    bash scripts/run_piper_replay_mock.sh none -- --action-horizon 8
EOF
}

MODE="local"
MODE_SET=0
KILL_EXISTING_REPLAY="${KILL_EXISTING_REPLAY:-1}"
REPLAY_KILL_GRACE_SEC="${REPLAY_KILL_GRACE_SEC:-5}"
REPLAY_MODE="${REPLAY_MODE:-policy}"
VISUALIZE="${VISUALIZE:-0}"
SAVE_PATH="${SAVE_PATH:-}"
MAIN_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    local|remote|none|mock)
      if [[ "$MODE_SET" == "1" ]]; then
        echo "Mode was specified more than once." >&2
        exit 2
      fi
      MODE="$1"
      MODE_SET=1
      shift
      ;;
    --mock-pi0)
      echo "Warning: --mock-pi0 is deprecated; use MODE=mock instead (e.g. 'bash $0 mock')." >&2
      if [[ "$MODE_SET" == "1" && "$MODE" != "mock" ]]; then
        echo "--mock-pi0 conflicts with MODE=$MODE." >&2
        exit 2
      fi
      MODE="mock"
      MODE_SET=1
      shift
      ;;
    --kill-existing-replay)
      KILL_EXISTING_REPLAY="1"
      shift
      ;;
    --planner-replay)
      REPLAY_MODE="planner"
      shift
      ;;
    --hybrid-replay)
      REPLAY_MODE="hybrid"
      shift
      ;;
    --policy-replay)
      REPLAY_MODE="policy"
      shift
      ;;
    --visualize)
      VISUALIZE=1
      shift
      ;;
    --save-path)
      if [[ "$#" -lt 2 ]]; then
        echo "--save-path requires a value." >&2
        exit 2
      fi
      SAVE_PATH="$2"
      shift 2
      ;;
    --save-path=*)
      SAVE_PATH="${1#*=}"
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

case "$REPLAY_MODE" in
  policy|planner|hybrid)
    ;;
  *)
    echo "REPLAY_MODE must be 'policy', 'planner', or 'hybrid'." >&2
    exit 2
    ;;
esac

server_require_nonnegative_int "REPLAY_KILL_GRACE_SEC" "$REPLAY_KILL_GRACE_SEC"

stop_existing_replay_processes() {
  if [[ "${KILL_EXISTING_REPLAY:-1}" != "1" ]]; then
    return 0
  fi

  local grace_sec="${REPLAY_KILL_GRACE_SEC:-5}"
  local -a replay_pids=()
  local -a remaining_pids=()
  local line pid proc_name
  local pgrep_pattern='\-m examples\.piper_real\.main .*--replay-dataset'
  if [[ "$REPLAY_MODE" == "planner" ]]; then
    pgrep_pattern='\-m examples\.piper_real\.main .*--replay-dataset .*--use-llm-planner'
  fi

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    pid="${line%% *}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    proc_name="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d '[:space:]')"
    [[ "$proc_name" == python* ]] || continue
    replay_pids+=("$pid")
    echo "Found existing replay mock process: $line"
  done < <(pgrep -af "$pgrep_pattern" || true)

  if [[ "${#replay_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "Stopping existing replay mock process(es): ${replay_pids[*]}"
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
        echo "Existing replay mock process(es) stopped."
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
    echo "Force killing replay mock process(es): ${remaining_pids[*]}"
    kill -KILL "${remaining_pids[@]}" 2>/dev/null || true
  fi
}

wait_for_pi0_ready() {
  local host="$1"
  local port="$2"

  if [[ "${WAIT_FOR_PI0_READY:-1}" != "1" ]]; then
    return 0
  fi

  local timeout_sec="${PI0_READY_TIMEOUT_SEC:-180}"
  local retry_interval_sec="${PI0_READY_RETRY_INTERVAL_SEC:-2}"
  local check_timeout_sec="${PI0_READY_CHECK_TIMEOUT_SEC:-5}"
  local deadline=$((SECONDS + timeout_sec))
  local attempt=1
  local check_mode="local"

  if [[ "$MODE" == "remote" ]]; then
    check_mode="remote"
  fi

  echo "Waiting for pi0 server at ws://$host:$port ..."
  while true; do
    if "$PYTHON_CMD" -m examples.piper_real.server_checks \
      --pi0-host "$host" \
      --pi0-port "$port" \
      --timeout-sec "$check_timeout_sec" \
      >/dev/null 2>&1; then
      echo "Pi0 server is ready: ws://$host:$port"
      return 0
    fi

    if (( SECONDS >= deadline )); then
      echo "Timed out after ${timeout_sec}s waiting for pi0 server: ws://$host:$port" >&2
      echo "Check status with: bash scripts/check_servers.sh $check_mode" >&2
      return 1
    fi

    echo "Pi0 not ready yet; retrying in ${retry_interval_sec}s (attempt ${attempt})..."
    attempt=$((attempt + 1))
    sleep "$retry_interval_sec"
  done
}

wait_for_planner_ready() {
  local base_url="$1"
  local model="$2"
  local api_key="${3:-}"

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
    local -a check_cmd=(
      "$PYTHON_CMD" -m examples.piper_real.server_checks
      --planner-base-url "$base_url" \
      --planner-model "$model" \
      --timeout-sec "$check_timeout_sec" \
    )
    if [[ -n "$api_key" ]]; then
      check_cmd+=(--planner-api-key "$api_key")
    fi

    if "${check_cmd[@]}" >/dev/null 2>&1; then
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

qz_state_value() {
  local key="$1"
  python3 -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
try:
    data = json.loads(path.read_text())
except FileNotFoundError:
    print(f"qz state file does not exist: {path}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as exc:
    print(f"invalid qz state file {path}: {exc}", file=sys.stderr)
    sys.exit(1)
value = data.get(key, "")
print(value)
' "$QZ_STATE_FILE" "$key"
}

print_command_redacted() {
  local redact_next=0
  local arg
  for arg in "$@"; do
    if [[ "$redact_next" == "1" ]]; then
      printf ' %q' "<redacted>"
      redact_next=0
      continue
    fi
    printf ' %q' "$arg"
    if [[ "$arg" == "--planner.api-key" ]]; then
      redact_next=1
    fi
  done
  printf '\n'
}

DATASET="${DATASET:-/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/aloha-data/long-horizon-demo/episode_4.hdf5}"
TASK_NAME="${TASK_NAME:-}"
DEFAULT_REPLAY_TASK_SPEC="$SERVER_REPO_ROOT/config/episode4_plate_wash_sandwich.task_spec.json"
REPLAY_TASK_SPEC="${REPLAY_TASK_SPEC:-${TASK_SPEC:-}}"
REPLAY_TASK_SPEC_AUTO_ENABLE=1
if [[ "${REPLAY_TASK_SPEC,,}" == "none" ]]; then
  REPLAY_TASK_SPEC=""
  REPLAY_TASK_SPEC_AUTO_ENABLE=0
fi
if [[ -z "$REPLAY_TASK_SPEC" && "$REPLAY_MODE" == "hybrid" && "$REPLAY_TASK_SPEC_AUTO_ENABLE" == "1" ]]; then
  if [[ "$(basename "$DATASET")" == "episode_4.hdf5" && -f "$DEFAULT_REPLAY_TASK_SPEC" ]]; then
    REPLAY_TASK_SPEC="$DEFAULT_REPLAY_TASK_SPEC"
  fi
fi
PROMPT_SOURCE="default"
if [[ -n "${PROMPT:-}" ]]; then
  PROMPT="$PROMPT"
  PROMPT_SOURCE="env:PROMPT"
elif [[ -n "$TASK_NAME" ]]; then
  PROMPT="$(server_lookup_task_prompt "$TASK_NAME")"
  PROMPT_SOURCE="task_catalog:$TASK_NAME"
elif [[ -n "$REPLAY_TASK_SPEC" ]]; then
  PROMPT="$(server_read_task_spec_total_task "$REPLAY_TASK_SPEC")"
  PROMPT_SOURCE="task_spec:$REPLAY_TASK_SPEC"
elif [[ "$REPLAY_MODE" == "planner" ]]; then
  PROMPT="long-horizon replay planner validation"
else
  PROMPT="Complete this long-horizon task in the exact order below. Phase 1, dish washing: pick up the center plate, turn on the faucet, wash the plate, turn off the faucet, and return the plate to its original position. Phase 2, transition: move from the sink area to the plate area. Phase 3, sandwich assembly: place the first bread slice on the center plate, place the lettuce on top of the bread on the center plate, and place the second bread slice on top of the lettuce. Do not skip, reorder, or revisit earlier subtasks unless the visual evidence clearly shows the current stage estimate is wrong."
fi
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
MANIPULATE_MAX_STEPS="${MANIPULATE_MAX_STEPS:-64}"
MANIPULATE_REPLAN_INTERVAL_STEPS="${MANIPULATE_REPLAN_INTERVAL_STEPS:-16}"
DEFAULT_PROGRESS_POLICY_CONFIG="pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3"
DEFAULT_PROGRESS_CHECKPOINT_DIR="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/openpi/checkpoints/pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3/pi05_pick_bread_leaf_progress_dual_20260424_061145/99999"
POLICY_CONFIG="${POLICY_CONFIG:-$DEFAULT_PROGRESS_POLICY_CONFIG}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_PROGRESS_CHECKPOINT_DIR}"
PROGRESS_SOURCE="${PROGRESS_SOURCE:-subtask}"
PROGRESS_HEAD_MODE="${PROGRESS_HEAD_MODE:-auto}"
NAVIGATION_ONLY="${NAVIGATION_ONLY:-1}"
PYTHON_CMD="${PYTHON_CMD:-$(server_default_python_cmd)}"
START_SERVERS="${START_SERVERS:-1}"
PLANNER_BACKEND="${PLANNER_BACKEND:-vllm}"
PLANNER_BACKEND="${PLANNER_BACKEND,,}"

if [[ "$REPLAY_MODE" == "planner" ]]; then
  START_TARGET="${START_TARGET:-planner}"
else
  START_TARGET="${START_TARGET:-pi0}"
fi

NEED_PI0=1
NEED_PLANNER=0
case "$REPLAY_MODE" in
  policy)
    NEED_PI0=1
    NEED_PLANNER=0
    ;;
  planner)
    NEED_PI0=0
    NEED_PLANNER=1
    ;;
  hybrid)
    NEED_PI0=1
    NEED_PLANNER=1
    ;;
esac

server_require_config

case "$PLANNER_BACKEND" in
  vllm|qz)
    ;;
  *)
    echo "PLANNER_BACKEND must be 'vllm' or 'qz'." >&2
    exit 2
    ;;
esac

if [[ "$REPLAY_MODE" == "planner" && "$MODE" == "mock" ]]; then
  echo "REPLAY_MODE=planner does not use pi0; MODE=mock has no effect. Use MODE=none/local/remote." >&2
  exit 2
fi

OPENPI_ROOT="${OPENPI_ROOT:-$(server_default_openpi_root)}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-}"

# if [[ -z "${PI0_HOST:-}" ]]; then
#   if [[ "$MODE" == "remote" ]]; then
#     PI0_HOST="$(server_cfg_optional pi0.remote.host)"
#     if [[ -z "$PI0_HOST" ]]; then
#       echo "PI0_HOST is required in remote mode. Set PI0_HOST or configure pi0.remote.host in config/servers.toml." >&2
#       exit 1
#     fi
#   else
if [[ "$NEED_PI0" == "1" ]]; then
  PI0_PORT="${PI0_PORT:-$(server_cfg pi0.port)}"
  if [[ -z "${PI0_HOST:-}" ]]; then
    if [[ "$MODE" == "remote" ]]; then
      echo "PI0_HOST is required in remote mode. Use the robot workstation reachable address, not the SSH alias." >&2
      exit 1
    fi
    PI0_HOST="127.0.0.1"
  fi
fi

server_require_nonnegative_int "MAX_EPISODE_STEPS" "$MAX_EPISODE_STEPS"
server_require_positive_int "MANIPULATE_MAX_STEPS" "$MANIPULATE_MAX_STEPS"
server_require_positive_int "MANIPULATE_REPLAN_INTERVAL_STEPS" "$MANIPULATE_REPLAN_INTERVAL_STEPS"
case "$PROGRESS_SOURCE" in
  task|subtask)
    ;;
  *)
    echo "PROGRESS_SOURCE must be one of: task, subtask." >&2
    exit 2
    ;;
esac
case "${PROGRESS_HEAD_MODE,,}" in
  auto|force|off)
    PROGRESS_HEAD_MODE="${PROGRESS_HEAD_MODE,,}"
    ;;
  *)
    echo "PROGRESS_HEAD_MODE must be one of: auto, force, off." >&2
    exit 2
    ;;
esac

PLANNER_BASE_URL=""
PLANNER_API_KEY=""
PLANNER_REPLANNER_ENABLE_THINKING="${PLANNER_REPLANNER_ENABLE_THINKING:-$(server_cfg_optional planner.manipulation_replanner_enable_thinking)}"
PLANNER_REPLANNER_MAX_TOKENS="${PLANNER_REPLANNER_MAX_TOKENS:-$(server_cfg_optional planner.manipulation_replanner_max_tokens)}"
if [[ "$NEED_PLANNER" == "1" ]]; then
  case "$PLANNER_BACKEND" in
    vllm)
      PLANNER_PORT="${PLANNER_PORT:-$(server_cfg vllm.port)}"

      if [[ -z "${PLANNER_HOST:-}" ]]; then
        if [[ "$MODE" == "remote" ]]; then
          PLANNER_HOST="$(server_cfg_optional vllm.remote.host)"
          if [[ -z "$PLANNER_HOST" ]]; then
            echo "PLANNER_HOST is required in remote mode for REPLAY_MODE=hybrid. Set PLANNER_HOST or configure vllm.remote.host in config/servers.toml." >&2
            exit 1
          fi
        elif [[ -n "${PI0_HOST:-}" ]]; then
          PLANNER_HOST="$PI0_HOST"
        elif [[ "$MODE" == "remote" ]]; then
          echo "PLANNER_HOST is required in remote mode for REPLAY_MODE=$REPLAY_MODE." >&2
          exit 1
        else
          PLANNER_HOST="127.0.0.1"
        fi
      fi

      if [[ -z "${PLANNER_MODEL:-}" ]]; then
        if [[ "$MODE" == "remote" ]]; then
          PLANNER_MODEL="$(server_cfg vllm.remote.served_model_name)"
        else
          PLANNER_MODEL="$(server_cfg vllm.local.served_model_name)"
        fi
      fi

      PLANNER_BASE_URL="http://$PLANNER_HOST:$PLANNER_PORT/v1"
      ;;
    qz)
      QZ_STATE_FILE="${QZ_STATE_FILE:-$(server_cfg_optional qz.state_file)}"
      QZ_STATE_FILE="${QZ_STATE_FILE:-$SERVER_REPO_ROOT/config/vllm_server_state.json}"
      if [[ "$QZ_STATE_FILE" != /* ]]; then
        QZ_STATE_FILE="$SERVER_REPO_ROOT/$QZ_STATE_FILE"
      fi
      QZ_SEQ="${QZ_SEQ:-$(qz_state_value seq)}"
      PLANNER_API_KEY="${PLANNER_API_KEY:-$(qz_state_value api_key)}"
      if [[ -z "$QZ_SEQ" || "$QZ_SEQ" == "0" ]]; then
        echo "qz seq is missing or zero in $QZ_STATE_FILE. Run scripts/qz_vllm_server.py --create first." >&2
        exit 1
      fi
      if [[ -z "$PLANNER_API_KEY" ]]; then
        echo "qz api_key is missing in $QZ_STATE_FILE." >&2
        exit 1
      fi
      PLANNER_MODEL="${PLANNER_MODEL:-$(server_cfg_optional qz.served_model_name)}"
      PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen3.5-9B}"
      QZ_ENDPOINT_TEMPLATE="${QZ_ENDPOINT_TEMPLATE:-$(server_cfg_optional qz.endpoint_template)}"
      if [[ -z "$QZ_ENDPOINT_TEMPLATE" ]]; then
        QZ_ENDPOINT_TEMPLATE='https://qwen35-9b-{seq}-inf.openapi-qb.sii.edu.cn/v1'
      fi
      if [[ -z "${PLANNER_BASE_URL:-}" ]]; then
        PLANNER_BASE_URL="$(python3 -c 'import sys; print(sys.argv[1].replace("{seq}", sys.argv[2]))' "$QZ_ENDPOINT_TEMPLATE" "$QZ_SEQ")"
      fi
      ;;
  esac
fi

PLANNER_REPLANNER_ENABLE_THINKING="$(server_normalize_bool "$PLANNER_REPLANNER_ENABLE_THINKING")"
NAVIGATION_ONLY="$(server_normalize_bool "$NAVIGATION_ONLY")"
VISUALIZE="$(server_normalize_bool "$VISUALIZE")"

if [[ -n "$PLANNER_REPLANNER_MAX_TOKENS" ]]; then
  server_require_positive_int "PLANNER_REPLANNER_MAX_TOKENS" "$PLANNER_REPLANNER_MAX_TOKENS"
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Replay dataset does not exist: $DATASET" >&2
  exit 1
fi

PYTHON_CMD="$(server_resolve_python_cmd "$PYTHON_CMD")"
OPENPI_CLIENT_SRC="$(server_resolve_openpi_client_src "$OPENPI_ROOT" "$OPENPI_CLIENT_SRC")"
server_export_openpi_pythonpath "$OPENPI_CLIENT_SRC"
export POLICY_CONFIG CHECKPOINT_DIR PROGRESS_SOURCE

if [[ "$START_TARGET" == "all" && "$REPLAY_MODE" == "policy" ]]; then
  echo "Note: START_TARGET=all only starts the planner server in addition to pi0."
  echo "This script still runs pi0 policy replay unless you pass REPLAY_MODE=planner/hybrid."
fi

cd "$SERVER_REPO_ROOT"
stop_existing_replay_processes

# MODE=mock: launch the built-in mock policy server instead of the real pi0 server.
# Honors PI0_HOST/PI0_PORT as set by the user (no silent overrides). The EXIT trap
# ensures the background mock process is cleaned up when the script exits.
MOCK_PI0_PID=""
if [[ "$MODE" == "mock" ]]; then
  echo "Starting mock policy server on ws://$PI0_HOST:$PI0_PORT ..."
  "$PYTHON_CMD" -m examples.piper_real.mock_policy_server \
    --host "$PI0_HOST" --port "$PI0_PORT" &
  MOCK_PI0_PID="$!"
  echo "Mock policy server PID: $MOCK_PI0_PID"

  trap 'if [[ -n "$MOCK_PI0_PID" ]] && kill -0 "$MOCK_PI0_PID" 2>/dev/null; then
    echo "Stopping mock policy server (PID $MOCK_PI0_PID)."
    kill "$MOCK_PI0_PID" 2>/dev/null || true
  fi' EXIT
fi

EFFECTIVE_START_TARGET="$START_TARGET"
if [[ "$REPLAY_MODE" == "hybrid" && "$START_SERVERS" == "1" && "$MODE" != "none" && "$MODE" != "mock" && "$EFFECTIVE_START_TARGET" == "pi0" ]]; then
  echo "REPLAY_MODE=hybrid requires planner + pi0; promoting START_TARGET=pi0 to all."
  EFFECTIVE_START_TARGET="all"
fi
if [[ "$NEED_PLANNER" == "1" && "$PLANNER_BACKEND" == "qz" && "$START_SERVERS" == "1" && "$MODE" != "none" && "$MODE" != "mock" && "$EFFECTIVE_START_TARGET" == "all" ]]; then
  if [[ "$NEED_PI0" == "1" ]]; then
    echo "PLANNER_BACKEND=qz uses an external planner; starting pi0 only."
    EFFECTIVE_START_TARGET="pi0"
  else
    echo "PLANNER_BACKEND=qz uses an external planner; skipping planner startup."
    START_SERVERS=0
  fi
fi

if [[ "$START_SERVERS" == "1" && "$MODE" != "none" && "$MODE" != "mock" ]]; then
  case "$EFFECTIVE_START_TARGET" in
    pi0)
      if [[ "$NEED_PI0" != "1" ]]; then
        echo "START_TARGET=pi0 is not applicable for REPLAY_MODE=$REPLAY_MODE." >&2
        exit 1
      fi
      if [[ "$MODE" == "local" ]]; then
        bash "$SCRIPT_DIR/start_pi0_server_local.sh"
      else
        bash "$SCRIPT_DIR/start_pi0_server.sh"
      fi
      ;;
    planner)
      if [[ "$NEED_PLANNER" != "1" ]]; then
        echo "START_TARGET=planner is not applicable for REPLAY_MODE=$REPLAY_MODE." >&2
        exit 1
      fi
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
      echo "Unsupported START_TARGET='$START_TARGET'. Expected 'pi0', 'planner', or 'all'." >&2
      exit 1
      ;;
  esac
else
  echo "Skipping server startup."
fi

if [[ "$NEED_PI0" == "1" ]]; then
  wait_for_pi0_ready "$PI0_HOST" "$PI0_PORT"
fi
if [[ "$NEED_PLANNER" == "1" ]]; then
  wait_for_planner_ready "$PLANNER_BASE_URL" "$PLANNER_MODEL" "$PLANNER_API_KEY"
fi

cmd=(
  "$PYTHON_CMD" -m examples.piper_real.main
  --prompt "$PROMPT"
  --replay-dataset "$DATASET"
  --replay-mode "$REPLAY_MODE"
  --max-episode-steps "$MAX_EPISODE_STEPS"
)

if [[ "$NEED_PI0" == "1" ]]; then
  cmd+=(--host "$PI0_HOST" --port "$PI0_PORT")
fi

if [[ "$REPLAY_MODE" == "planner" ]]; then
  cmd+=(
    --use-llm-planner
    --planner.base-url "$PLANNER_BASE_URL"
    --planner.model "$PLANNER_MODEL"
  )
  if [[ -n "$PLANNER_API_KEY" ]]; then
    cmd+=(--planner.api-key "$PLANNER_API_KEY")
  fi
  if [[ "$NAVIGATION_ONLY" == "1" ]]; then
    cmd+=(--navigation-only)
  fi
fi

if [[ "$REPLAY_MODE" == "hybrid" ]]; then
  cmd+=(
    --use-llm-planner
    --planner.base-url "$PLANNER_BASE_URL"
    --planner.model "$PLANNER_MODEL"
    --replay-manipulate-max-steps "$MANIPULATE_MAX_STEPS"
    --replay-manipulate-replan-interval-steps "$MANIPULATE_REPLAN_INTERVAL_STEPS"
    --progress-head-mode "$PROGRESS_HEAD_MODE"
  )
  if [[ -n "$PLANNER_API_KEY" ]]; then
    cmd+=(--planner.api-key "$PLANNER_API_KEY")
  fi
  if [[ -n "$PLANNER_REPLANNER_ENABLE_THINKING" ]]; then
    if [[ "$PLANNER_REPLANNER_ENABLE_THINKING" == "1" ]]; then
      cmd+=(--planner.manipulation-replanner-enable-thinking)
    else
      cmd+=(--planner.no-manipulation-replanner-enable-thinking)
    fi
  fi
  if [[ -n "$PLANNER_REPLANNER_MAX_TOKENS" ]]; then
    cmd+=(--planner.manipulation-replanner-max-tokens "$PLANNER_REPLANNER_MAX_TOKENS")
  fi
  if [[ -n "$REPLAY_TASK_SPEC" ]]; then
    cmd+=(--planner.task-spec-path "$REPLAY_TASK_SPEC")
  fi
fi

if [[ "$VISUALIZE" == "1" ]]; then
  cmd+=(--visualize)
fi

if [[ -n "$SAVE_PATH" ]]; then
  cmd+=(--save-path "$SAVE_PATH")
fi

if [[ "${#MAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${MAIN_ARGS[@]}")
fi

echo "Mode: $MODE"
echo "Replay mode: $REPLAY_MODE"
if [[ "$NEED_PI0" == "1" ]]; then
  echo "PI0 host: $PI0_HOST"
  echo "PI0 port: $PI0_PORT"
  echo "Policy config: $POLICY_CONFIG"
  echo "Checkpoint: $CHECKPOINT_DIR"
  echo "Progress source: $PROGRESS_SOURCE"
fi
echo "Dataset: $DATASET"
echo "Task name: ${TASK_NAME:-<unset>}"
echo "Prompt: $PROMPT"
echo "Prompt source: $PROMPT_SOURCE"
echo "Max episode steps: $MAX_EPISODE_STEPS"
if [[ "$NEED_PLANNER" == "1" ]]; then
  echo "Planner backend: $PLANNER_BACKEND"
  echo "Planner base URL: $PLANNER_BASE_URL"
  echo "Planner model: $PLANNER_MODEL"
  if [[ "$PLANNER_BACKEND" == "qz" ]]; then
    echo "qz state file: $QZ_STATE_FILE"
  fi
fi
if [[ "$REPLAY_MODE" == "planner" ]]; then
  echo "Navigation only: $NAVIGATION_ONLY"
fi
if [[ "$REPLAY_MODE" == "hybrid" ]]; then
  echo "Manipulate max steps: $MANIPULATE_MAX_STEPS"
  echo "Manipulate replan interval steps: $MANIPULATE_REPLAN_INTERVAL_STEPS"
  echo "Progress head mode: $PROGRESS_HEAD_MODE"
  echo "Manipulation replanner enable thinking: ${PLANNER_REPLANNER_ENABLE_THINKING:-<default>}"
  echo "Manipulation replanner max tokens: ${PLANNER_REPLANNER_MAX_TOKENS:-<default>}"
  echo "Replay task spec: ${REPLAY_TASK_SPEC:-<unset>}"
fi
echo "Visualize: $VISUALIZE"
echo "Save path: ${SAVE_PATH:-<unset>}"
echo "Python: $PYTHON_CMD"
echo "openpi_client src: $OPENPI_CLIENT_SRC"
echo "Start target: $EFFECTIVE_START_TARGET"
echo "Running:"
print_command_redacted "${cmd[@]}"

"${cmd[@]}"
