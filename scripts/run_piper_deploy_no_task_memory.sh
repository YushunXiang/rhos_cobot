#!/usr/bin/env bash
# Real-robot hybrid deployment with ordered task memory disabled.
#
# This wraps scripts/run_piper_deploy.sh for the fallback path where the
# outer task decomposition still runs, but --planner.task-spec-path is not
# passed to examples.piper_real.main. That prevents stale ordered task memory
# from blocking manipulation subtask completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

PROMPT_SOURCE="${PROMPT_SOURCE:-$REPO_ROOT/config/deploy.json}"
if [[ -z "${PROMPT:-}" ]]; then
  if [[ ! -f "$PROMPT_SOURCE" ]]; then
    echo "PROMPT is unset and prompt source does not exist: $PROMPT_SOURCE" >&2
    exit 1
  fi
  PROMPT="$(
    python3 -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_task"])' \
      "$PROMPT_SOURCE"
  )"
fi

LOG="${LOG:-logs/deploy_hybrid_no_task_memory_$(date +%Y%m%d_%H%M%S).log}"

export PROMPT
export TASK_SPEC=none
export MODE="${MODE:-hybrid}"
export ROBOT_BASE_TOPIC="${ROBOT_BASE_TOPIC:-/odom}"
export no_proxy='*'
export NO_PROXY='*'

echo "Running hybrid deploy with ordered task memory disabled."
echo "Prompt source: $PROMPT_SOURCE"
echo "Robot base odom topic: $ROBOT_BASE_TOPIC"
echo "Log: $LOG"

bash scripts/run_piper_deploy.sh 2>&1 | tee "$LOG"
