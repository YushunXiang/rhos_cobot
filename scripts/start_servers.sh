#!/usr/bin/env bash
# Start both the vLLM planner server and the Pi0 policy server.
#
# Usage:
#   bash scripts/start_servers.sh              # local mode (default)
#   bash scripts/start_servers.sh local         # local mode
#   bash scripts/start_servers.sh remote        # remote mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/config/servers.toml"

_cfg() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1"; }

MODE="${1:-local}"

kill_local_sessions() {
  local vllm_session pi0_session
  vllm_session="$(_cfg vllm.local.session_name)"
  pi0_session="$(_cfg pi0.local.session_name)"
  echo "Killing local tmux sessions: $vllm_session, $pi0_session"
  tmux kill-session -t "$vllm_session" 2>/dev/null || true
  tmux kill-session -t "$pi0_session" 2>/dev/null || true
}

kill_remote_sessions() {
  local remote_host session
  remote_host="$(_cfg vllm.remote.host)"
  session="$(_cfg vllm.remote.session_name)"
  echo "Killing remote tmux session: $session on $remote_host"
  ssh "$remote_host" "tmux kill-session -t $session 2>/dev/null || true"

  remote_host="$(_cfg pi0.remote.host)"
  session="$(_cfg pi0.remote.session_name)"
  echo "Killing remote tmux session: $session on $remote_host"
  ssh "$remote_host" "tmux kill-session -t $session 2>/dev/null || true"
}

case "$MODE" in
  local)
    echo "=== Starting local servers ==="
    echo ""
    kill_local_sessions
    echo ""
    echo "--- vLLM planner server ---"
    bash "$SCRIPT_DIR/start_vllm_server_local.sh"
    echo ""
    echo "--- Pi0 policy server ---"
    bash "$SCRIPT_DIR/start_pi0_server_local.sh"
    ;;
  remote)
    echo "=== Starting remote servers ==="
    echo ""
    kill_remote_sessions
    echo ""
    echo "--- vLLM planner server ---"
    bash "$SCRIPT_DIR/start_vllm_server.sh"
    echo ""
    echo "--- Pi0 policy server ---"
    bash "$SCRIPT_DIR/start_pi0_server.sh"
    ;;
  *)
    echo "Usage: bash scripts/start_servers.sh [local|remote]" >&2
    exit 1
    ;;
esac

echo ""
echo "=== Both servers started ==="
