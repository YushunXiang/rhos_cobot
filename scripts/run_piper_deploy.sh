#!/usr/bin/env bash
# Real-robot deployment for examples.piper_real.main.
#
# Two deployment modes:
#   planner  Navigation-only (VLM planner drives the mobile base; no pi0 required)
#   hybrid   VLM planner navigation + pi0 manipulation (long-horizon tasks)
#
# Usage:
#   bash scripts/run_piper_deploy.sh                    # hybrid (default)
#   MODE=planner PROMPT="turn on the tap." \
#       bash scripts/run_piper_deploy.sh
#   MODE=hybrid TASK_NAME=plate_wash_sandwich \
#       TASK_SPEC=config/episode4_plate_wash_sandwich.task_spec.json \
#       bash scripts/run_piper_deploy.sh
#
# Env overrides:
#   PROMPT             Total-task narrative forwarded as --prompt.
#   TASK_NAME          If PROMPT is unset, resolve prompt from config/task_prompts.json.
#   TASK_SPEC          Ordered task-spec JSON (hybrid only); forwarded as
#                      --planner.task-spec-path for strict subtask ordering.
#                      Set to 'none' to disable the default config/deploy.json.
#   PI0_HOST/PI0_PORT  pi0 policy server (hybrid mode). Defaults from config/servers.toml.
#   PLANNER_HOST/PORT  vLLM planner server. Defaults from config/servers.toml.
#   PLANNER_MODEL      Planner model id. Defaults from config/servers.toml.
#   PYTHON_CMD         Python interpreter. Default examples/piper_real/.venv/bin/python,
#                      fallback python3.
#   NAVIGATION_ONLY    Pass --navigation-only in planner mode (default 1).
#   MANIPULATE_MAX_STEPS               Hybrid: per-subtask policy step cap (default 10000).
#   MANIPULATE_REPLAN_INTERVAL_STEPS   Hybrid: VLM replan every N policy steps (default 100).
#   PLANNER_REPLANNER_ENABLE_THINKING  Hybrid: 0/1 to toggle replanner thinking (default: server config).
#   PLANNER_REPLANNER_MAX_TOKENS       Hybrid: override replanner max tokens (default: server config).
#   MAX_EPISODE_STEPS  Hard cap per manipulate invocation (default 0 = disabled).
#   ROBOT_BASE_TOPIC   Odometry topic used by deploy navigation (default /odom_raw).
#                      Set ROBOT_BASE_TOPIC=/odom if tracer_bringup publishes /odom.
#   ROBOT_BASE_CMD_TOPIC                Base velocity command topic (default /cmd_vel).
#   WAIT_FOR_ROBOT_BASE_ODOM           0/1 preflight odom availability before deploy (default 1).
#   ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC   Preflight odom wait timeout seconds (default 10).
#   VISUALIZE          Pass --visualize (default 0; rarely useful on real robot).
#   SAVE_PATH          Optional --save-path MP4 output (default unset).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

REPO_ROOT="$SERVER_REPO_ROOT"
cd "$REPO_ROOT"

MODE="${MODE:-hybrid}"
server_require_config

PYTHON_CMD="${PYTHON_CMD:-$(server_default_python_cmd)}"
PYTHON_CMD="$(server_resolve_python_cmd "$PYTHON_CMD")"
NAVIGATION_ONLY="${NAVIGATION_ONLY:-1}"
DEFAULT_DEPLOY_SPEC="$REPO_ROOT/config/deploy.json"
TASK_SPEC="${TASK_SPEC:-$DEFAULT_DEPLOY_SPEC}"
MANIPULATE_MAX_STEPS="${MANIPULATE_MAX_STEPS:-10000}"
MANIPULATE_REPLAN_INTERVAL_STEPS="${MANIPULATE_REPLAN_INTERVAL_STEPS:-100}"
PLANNER_REPLANNER_ENABLE_THINKING="${PLANNER_REPLANNER_ENABLE_THINKING:-$(server_cfg_optional planner.manipulation_replanner_enable_thinking)}"
PLANNER_REPLANNER_MAX_TOKENS="${PLANNER_REPLANNER_MAX_TOKENS:-$(server_cfg_optional planner.manipulation_replanner_max_tokens)}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
ROBOT_BASE_TOPIC="${ROBOT_BASE_TOPIC:-/odom_raw}"
ROBOT_BASE_CMD_TOPIC="${ROBOT_BASE_CMD_TOPIC:-/cmd_vel}"
WAIT_FOR_ROBOT_BASE_ODOM="${WAIT_FOR_ROBOT_BASE_ODOM:-1}"
ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC="${ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC:-10}"
VISUALIZE="${VISUALIZE:-0}"
SAVE_PATH="${SAVE_PATH:-}"

PI0_PORT="${PI0_PORT:-$(server_cfg pi0.port)}"
PI0_HOST="${PI0_HOST:-$(server_cfg_optional pi0.remote.host)}"
PLANNER_PORT="${PLANNER_PORT:-$(server_cfg vllm.port)}"
PLANNER_HOST="${PLANNER_HOST:-$(server_cfg_optional vllm.remote.host)}"
if [[ -z "$PLANNER_HOST" ]]; then
  PLANNER_HOST="127.0.0.1"
fi
PLANNER_MODEL="${PLANNER_MODEL:-$(server_cfg_optional vllm.remote.served_model_name)}"
if [[ -z "$PLANNER_MODEL" ]]; then
  PLANNER_MODEL="$(server_cfg_optional vllm.local.served_model_name)"
fi
if [[ -z "$PLANNER_MODEL" ]]; then
  PLANNER_MODEL="Qwen/Qwen3.5-4B"
fi
PLANNER_BASE_URL="http://${PLANNER_HOST}:${PLANNER_PORT}/v1"

if [[ "${TASK_SPEC,,}" == "none" ]]; then
  TASK_SPEC=""
fi

if [[ -n "$TASK_SPEC" && ! -f "$TASK_SPEC" ]]; then
  echo "TASK_SPEC file does not exist: $TASK_SPEC" >&2
  exit 1
fi

NAVIGATION_ONLY="$(server_normalize_bool "$NAVIGATION_ONLY")"
VISUALIZE="$(server_normalize_bool "$VISUALIZE")"
WAIT_FOR_ROBOT_BASE_ODOM="$(server_normalize_bool "$WAIT_FOR_ROBOT_BASE_ODOM")"
PLANNER_REPLANNER_ENABLE_THINKING="$(server_normalize_bool "$PLANNER_REPLANNER_ENABLE_THINKING")"
server_require_positive_int "MANIPULATE_MAX_STEPS" "$MANIPULATE_MAX_STEPS"
server_require_positive_int "MANIPULATE_REPLAN_INTERVAL_STEPS" "$MANIPULATE_REPLAN_INTERVAL_STEPS"
server_require_nonnegative_int "MAX_EPISODE_STEPS" "$MAX_EPISODE_STEPS"
server_require_positive_int "ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC" "$ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC"
if [[ -n "$PLANNER_REPLANNER_MAX_TOKENS" ]]; then
  server_require_positive_int "PLANNER_REPLANNER_MAX_TOKENS" "$PLANNER_REPLANNER_MAX_TOKENS"
fi

if [[ -z "$ROBOT_BASE_TOPIC" ]]; then
  echo "ROBOT_BASE_TOPIC must be non-empty." >&2
  exit 1
fi

if [[ -z "$ROBOT_BASE_CMD_TOPIC" ]]; then
  echo "ROBOT_BASE_CMD_TOPIC must be non-empty." >&2
  exit 1
fi

wait_for_robot_base_odom() {
  if [[ "$WAIT_FOR_ROBOT_BASE_ODOM" != "1" ]]; then
    return 0
  fi
  if ! command -v rostopic >/dev/null 2>&1; then
    echo "rostopic command not found; cannot preflight $ROBOT_BASE_TOPIC." >&2
    echo "Source the ROS environment first, or set WAIT_FOR_ROBOT_BASE_ODOM=0 to skip this check." >&2
    exit 1
  fi

  echo "Waiting for odometry on $ROBOT_BASE_TOPIC (timeout ${ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC}s)..."
  if ! timeout "$ROBOT_BASE_ODOM_WAIT_TIMEOUT_SEC" rostopic echo -n1 "$ROBOT_BASE_TOPIC" >/dev/null 2>&1; then
    echo "ERROR: no odometry data on $ROBOT_BASE_TOPIC." >&2
    echo "Start roscore + tracer_bringup first. If the base publishes /odom, rerun with ROBOT_BASE_TOPIC=/odom." >&2
    exit 1
  fi
  echo "Odometry OK: $ROBOT_BASE_TOPIC"
}

# --- Resolve prompt ---
# Precedence: PROMPT env > TASK_NAME lookup > total_task from TASK_SPEC.
if [[ -z "${PROMPT:-}" && -n "${TASK_NAME:-}" ]]; then
  PROMPT="$(server_lookup_task_prompt "$TASK_NAME")"
fi
if [[ -z "${PROMPT:-}" && -n "$TASK_SPEC" && -f "$TASK_SPEC" ]]; then
  PROMPT="$(server_read_task_spec_total_task "$TASK_SPEC")"
fi
: "${PROMPT:?Set PROMPT, TASK_NAME, or provide a TASK_SPEC with total_task}"

if [[ "$MODE" == "hybrid" && -z "$PI0_HOST" ]]; then
  echo "PI0_HOST is required in hybrid mode. Set PI0_HOST or configure pi0.remote.host in config/servers.toml." >&2
  exit 1
fi

cmd=(
  "$PYTHON_CMD" -B -u -m examples.piper_real.main
  --prompt "$PROMPT"
  --use-llm-planner
  --use-robot-base
  --robot-base-topic "$ROBOT_BASE_TOPIC"
  --robot-base-cmd-topic "$ROBOT_BASE_CMD_TOPIC"
  --planner.base-url "$PLANNER_BASE_URL"
  --planner.model    "$PLANNER_MODEL"
)

case "$MODE" in
  planner)
    if [[ "$NAVIGATION_ONLY" == "1" ]]; then
      cmd+=(--navigation-only)
    fi
    # TASK_SPEC is only used in hybrid mode; silently ignore here.
    ;;
  hybrid)
    cmd+=(--host "$PI0_HOST" --port "$PI0_PORT")
    cmd+=(
      --replay-manipulate-max-steps "$MANIPULATE_MAX_STEPS"
      --replay-manipulate-replan-interval-steps "$MANIPULATE_REPLAN_INTERVAL_STEPS"
    )
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
    cmd+=(--max-episode-steps "$MAX_EPISODE_STEPS")
    if [[ -n "$TASK_SPEC" ]]; then
      cmd+=(--planner.task-spec-path "$TASK_SPEC")
    fi
    ;;
  *)
    echo "Unknown MODE='$MODE'. Expected 'planner' or 'hybrid'." >&2
    exit 2
    ;;
esac

if [[ "$VISUALIZE" == "1" ]]; then
  cmd+=(--visualize)
fi

if [[ -n "$SAVE_PATH" ]]; then
  cmd+=(--save-path "$SAVE_PATH")
fi

echo "Mode: $MODE"
echo "Planner: $PLANNER_BASE_URL ($PLANNER_MODEL)"
if [[ "$MODE" == "hybrid" ]]; then
  echo "pi0: ws://$PI0_HOST:$PI0_PORT"
  echo "Task spec: ${TASK_SPEC:-<unset>}"
  echo "Manipulate max steps: $MANIPULATE_MAX_STEPS"
  echo "Manipulate replan interval steps: $MANIPULATE_REPLAN_INTERVAL_STEPS"
  echo "Manipulation replanner enable thinking: ${PLANNER_REPLANNER_ENABLE_THINKING:-<default>}"
  echo "Manipulation replanner max tokens: ${PLANNER_REPLANNER_MAX_TOKENS:-<default>}"
  echo "Max episode steps: $MAX_EPISODE_STEPS"
fi
echo "Visualize: $VISUALIZE"
echo "Save path: ${SAVE_PATH:-<unset>}"
echo "Python: $PYTHON_CMD"
echo "Robot base odom topic: $ROBOT_BASE_TOPIC"
echo "Robot base cmd topic: $ROBOT_BASE_CMD_TOPIC"
echo "Wait for robot base odom: $WAIT_FOR_ROBOT_BASE_ODOM"
echo "Prompt: $PROMPT"
echo "Running:"
printf '  %q' "${cmd[@]}"
printf '\n'

wait_for_robot_base_odom

"${cmd[@]}"
