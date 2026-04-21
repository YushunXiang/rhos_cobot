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
#   PI0_HOST/PI0_PORT  pi0 policy server (hybrid mode). Default 192.168.3.101:8001.
#   PLANNER_HOST/PORT  vLLM planner server. Default 192.168.3.123:8000.
#   PLANNER_MODEL      Planner model id. Default Qwen/Qwen3.5-4B.
#   PYTHON_CMD         Python interpreter. Default examples/piper_real/.venv/bin/python.
#   NAVIGATION_ONLY    Pass --navigation-only in planner mode (default 1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-hybrid}"
PI0_HOST="${PI0_HOST:-192.168.3.101}"
PI0_PORT="${PI0_PORT:-8001}"
PLANNER_HOST="${PLANNER_HOST:-192.168.3.110}"
PLANNER_PORT="${PLANNER_PORT:-8000}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen3.5-4B}"
PLANNER_BASE_URL="http://${PLANNER_HOST}:${PLANNER_PORT}/v1"
PYTHON_CMD="${PYTHON_CMD:-examples/piper_real/.venv/bin/python}"
NAVIGATION_ONLY="${NAVIGATION_ONLY:-1}"
DEFAULT_DEPLOY_SPEC="$REPO_ROOT/config/deploy.json"
TASK_SPEC="${TASK_SPEC:-$DEFAULT_DEPLOY_SPEC}"

if [[ ! -x "$PYTHON_CMD" ]]; then
  echo "PYTHON_CMD not executable: $PYTHON_CMD" >&2
  echo "Tip: activate the piper_real venv or set PYTHON_CMD." >&2
  exit 1
fi

if [[ -n "$TASK_SPEC" && ! -f "$TASK_SPEC" ]]; then
  echo "TASK_SPEC file does not exist: $TASK_SPEC" >&2
  exit 1
fi

# --- Resolve prompt ---
# Precedence: PROMPT env > TASK_NAME lookup > total_task from TASK_SPEC.
if [[ -z "${PROMPT:-}" && -n "${TASK_NAME:-}" ]]; then
  PROMPT="$("$PYTHON_CMD" "$SCRIPT_DIR/_resolve_task_prompt.py" \
              "$REPO_ROOT/config/task_prompts.json" "$TASK_NAME")"
fi
if [[ -z "${PROMPT:-}" && -n "$TASK_SPEC" && -f "$TASK_SPEC" ]]; then
  PROMPT="$("$PYTHON_CMD" -c '
import json, sys
print(json.load(open(sys.argv[1]))["total_task"])
' "$TASK_SPEC")"
fi
: "${PROMPT:?Set PROMPT, TASK_NAME, or provide a TASK_SPEC with total_task}"

cmd=(
  "$PYTHON_CMD" -m examples.piper_real.main
  --prompt "$PROMPT"
  --use-llm-planner
  --use-robot-base
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
    if [[ -n "$TASK_SPEC" ]]; then
      cmd+=(--planner.task-spec-path "$TASK_SPEC")
    fi
    ;;
  *)
    echo "Unknown MODE='$MODE'. Expected 'planner' or 'hybrid'." >&2
    exit 2
    ;;
esac

echo "Mode: $MODE"
echo "Planner: $PLANNER_BASE_URL ($PLANNER_MODEL)"
if [[ "$MODE" == "hybrid" ]]; then
  echo "pi0: ws://$PI0_HOST:$PI0_PORT"
  echo "Task spec: ${TASK_SPEC:-<unset>}"
fi
echo "Prompt: $PROMPT"
echo "Running:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
