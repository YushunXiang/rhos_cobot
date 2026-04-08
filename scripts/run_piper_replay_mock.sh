#!/usr/bin/env bash
# Run examples.piper_real.main in server-backed replay mock mode.
#
# Usage:
#   bash scripts/run_piper_replay_mock.sh
#   bash scripts/run_piper_replay_mock.sh local
#   PI0_HOST=192.168.3.101 bash scripts/run_piper_replay_mock.sh remote
#   START_SERVERS=0 PYTHON_CMD=../openpi/.venv/bin/python bash scripts/run_piper_replay_mock.sh none --action-horizon 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/run_piper_replay_mock.sh [wrapper options] [local|remote|none] [--] [extra main.py args...]

Modes:
  local   Start the Pi0 policy server locally (default)
  remote  Start the Pi0 policy server remotely
  none    Do not start servers; connect to an already running pi0 policy server

Wrapper options:
  --planner-replay    Delegate to offline VLM planner replay instead of pi0 policy replay
  --policy-replay     Force pi0 policy replay
  --kill-existing-replay
                     Stop any existing replay mock process before starting
                     default: enabled
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
                     values: policy, planner
                     default: policy
  MAX_EPISODE_STEPS  --max-episode-steps for replay
                     default: 0
  PI0_HOST           pi0 policy server host
                     default: 127.0.0.1 for local/none; required for remote
  PI0_PORT           pi0 policy server port
                     default: value from config/servers.toml
  OPENPI_ROOT        Local OpenPI repo root used to resolve openpi_client
                     default: value from config/servers.toml -> pi0.local.openpi_root
  PYTHON_CMD         Python interpreter used to run main.py
                     default: examples/piper_real/.venv/bin/python, fallback: python3
  START_SERVERS      Set to 0 to skip server startup even in local/remote mode
                     default: 1
  START_TARGET       Which startup helper to use in local/remote mode
                     values: pi0, all
                     default: pi0
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

Examples:
  bash scripts/run_piper_replay_mock.sh
  bash scripts/run_piper_replay_mock.sh --replay-kill-grace-sec 10
  bash scripts/run_piper_replay_mock.sh --no-kill-existing-replay none
  REPLAY_MODE=planner START_TARGET=all bash scripts/run_piper_replay_mock.sh
  PI0_HOST=192.168.3.101 bash scripts/run_piper_replay_mock.sh remote
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
    --planner-replay)
      REPLAY_MODE="planner"
      shift
      ;;
    --policy-replay)
      REPLAY_MODE="policy"
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
  policy|planner)
    ;;
  *)
    echo "REPLAY_MODE must be 'policy' or 'planner'." >&2
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
    echo "Found existing replay mock process: $line"
  done < <(pgrep -af '\-m examples\.piper_real\.main .*--replay-dataset' || true)

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

delegate_to_planner_replay() {
  local planner_start_target="$START_TARGET"
  case "$planner_start_target" in
    pi0)
      planner_start_target="planner"
      ;;
    all|planner)
      ;;
    *)
      echo "Unsupported START_TARGET='$START_TARGET' for planner replay. Expected 'pi0', 'planner', or 'all'." >&2
      exit 1
      ;;
  esac

  if [[ -n "${PI0_HOST:-}" && -z "${PLANNER_HOST:-}" ]]; then
    export PLANNER_HOST="$PI0_HOST"
  fi
  if [[ -n "${PI0_PORT:-}" && -z "${PLANNER_PORT:-}" ]]; then
    export PLANNER_PORT="$PI0_PORT"
  fi
  if [[ -n "${WAIT_FOR_PI0_READY:-}" && -z "${WAIT_FOR_PLANNER_READY:-}" ]]; then
    export WAIT_FOR_PLANNER_READY="$WAIT_FOR_PI0_READY"
  fi
  if [[ -n "${PI0_READY_TIMEOUT_SEC:-}" && -z "${PLANNER_READY_TIMEOUT_SEC:-}" ]]; then
    export PLANNER_READY_TIMEOUT_SEC="$PI0_READY_TIMEOUT_SEC"
  fi
  if [[ -n "${PI0_READY_RETRY_INTERVAL_SEC:-}" && -z "${PLANNER_READY_RETRY_INTERVAL_SEC:-}" ]]; then
    export PLANNER_READY_RETRY_INTERVAL_SEC="$PI0_READY_RETRY_INTERVAL_SEC"
  fi
  if [[ -n "${PI0_READY_CHECK_TIMEOUT_SEC:-}" && -z "${PLANNER_READY_CHECK_TIMEOUT_SEC:-}" ]]; then
    export PLANNER_READY_CHECK_TIMEOUT_SEC="$PI0_READY_CHECK_TIMEOUT_SEC"
  fi

  export DATASET PROMPT MAX_EPISODE_STEPS PYTHON_CMD START_SERVERS OPENPI_ROOT OPENPI_CLIENT_SRC
  export TASK_NAME PROMPT_SOURCE
  export KILL_EXISTING_REPLAY REPLAY_KILL_GRACE_SEC
  export START_TARGET="$planner_start_target"

  echo "REPLAY_MODE=planner detected; delegating to scripts/run_piper_replay_planner.sh"
  local -a delegate_cmd=(bash "$SCRIPT_DIR/run_piper_replay_planner.sh" "$MODE")
  if [[ "${#MAIN_ARGS[@]}" -gt 0 ]]; then
    delegate_cmd+=(-- "${MAIN_ARGS[@]}")
  fi
  printf '  %q' "${delegate_cmd[@]}"
  printf '\n'
  exec "${delegate_cmd[@]}"
}

DATASET="${DATASET:-/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/aloha-data/long-horizon-demo/episode_4.hdf5}"
TASK_NAME="${TASK_NAME:-}"
PROMPT_SOURCE="default"
if [[ -n "${PROMPT:-}" ]]; then
  PROMPT="$PROMPT"
  PROMPT_SOURCE="env:PROMPT"
elif [[ -n "$TASK_NAME" ]]; then
  PROMPT="$(server_lookup_task_prompt "$TASK_NAME")"
  PROMPT_SOURCE="task_catalog:$TASK_NAME"
else
  PROMPT="Analyze the video of a robotic arm performing sequential kitchen tasks and break down its actions into discrete subtasks. In the first phase, the robot grasps a white empty plate, positions it over a stainless steel sink, and rinses it under running water from the faucet. In the second phase, the scene shifts to a countertop setup with three plates: one with lettuce on the left, an empty plate in the middle, and one with two slices of bread on the right. Document the sandwich assembly process where the robot sequentially picks up one slice of bread from the right plate and places it onto the middle plate, transfers a piece of lettuce from the left plate onto the bread, and finally grasps the second slice of bread from the right plate, moving it to cover the lettuce."
fi
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
PYTHON_CMD="${PYTHON_CMD:-$(server_default_python_cmd)}"
START_SERVERS="${START_SERVERS:-1}"
START_TARGET="${START_TARGET:-pi0}"

server_require_config

if [[ "$REPLAY_MODE" == "planner" ]]; then
  delegate_to_planner_replay
fi

PI0_PORT="${PI0_PORT:-$(server_cfg pi0.port)}"
OPENPI_ROOT="${OPENPI_ROOT:-$(server_default_openpi_root)}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-}"

if [[ -z "${PI0_HOST:-}" ]]; then
  if [[ "$MODE" == "remote" ]]; then
    echo "PI0_HOST is required in remote mode. Use the robot workstation reachable address, not the SSH alias." >&2
    exit 1
  fi
  PI0_HOST="127.0.0.1"
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Replay dataset does not exist: $DATASET" >&2
  exit 1
fi

PYTHON_CMD="$(server_resolve_python_cmd "$PYTHON_CMD")"
OPENPI_CLIENT_SRC="$(server_resolve_openpi_client_src "$OPENPI_ROOT" "$OPENPI_CLIENT_SRC")"
server_export_openpi_pythonpath "$OPENPI_CLIENT_SRC"

if [[ "$START_TARGET" == "all" ]]; then
  echo "Note: START_TARGET=all only starts the planner server in addition to pi0."
  echo "This script still runs pi0 policy replay unless you pass REPLAY_MODE=planner or --planner-replay."
fi

cd "$SERVER_REPO_ROOT"
stop_existing_replay_processes

if [[ "$START_SERVERS" == "1" && "$MODE" != "none" ]]; then
  case "$START_TARGET" in
    pi0)
      if [[ "$MODE" == "local" ]]; then
        bash "$SCRIPT_DIR/start_pi0_server_local.sh"
      else
        bash "$SCRIPT_DIR/start_pi0_server.sh"
      fi
      ;;
    all)
      bash "$SCRIPT_DIR/start_servers.sh" "$MODE"
      ;;
    *)
      echo "Unsupported START_TARGET='$START_TARGET'. Expected 'pi0' or 'all'." >&2
      exit 1
      ;;
  esac
else
  echo "Skipping server startup."
fi

wait_for_pi0_ready "$PI0_HOST" "$PI0_PORT"

cmd=(
  "$PYTHON_CMD" -m examples.piper_real.main
  --host "$PI0_HOST"
  --port "$PI0_PORT"
  --prompt "$PROMPT"
  --replay-dataset "$DATASET"
  --max-episode-steps "$MAX_EPISODE_STEPS"
)

if [[ "${#MAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${MAIN_ARGS[@]}")
fi

echo "Mode: $MODE"
echo "PI0 host: $PI0_HOST"
echo "PI0 port: $PI0_PORT"
echo "Dataset: $DATASET"
echo "Task name: ${TASK_NAME:-<unset>}"
echo "Prompt: $PROMPT"
echo "Prompt source: $PROMPT_SOURCE"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Python: $PYTHON_CMD"
echo "openpi_client src: $OPENPI_CLIENT_SRC"
echo "Start target: $START_TARGET"
echo "Running:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
