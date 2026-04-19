#!/usr/bin/env bash
# Start both the vLLM planner server and the Pi0 policy server.
#
# Usage:
#   bash scripts/start_servers.sh              # local mode (default)
#   bash scripts/start_servers.sh local         # local mode
#   bash scripts/start_servers.sh remote        # remote mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

MODE="${1:-local}"
server_require_config

kill_local_sessions() {
  local vllm_session pi0_session
  vllm_session="$(server_cfg vllm.local.session_name)"
  pi0_session="$(server_cfg pi0.local.session_name)"
  echo "Killing local tmux sessions: $vllm_session, $pi0_session"
  tmux kill-session -t "$vllm_session" 2>/dev/null || true
  tmux kill-session -t "$pi0_session" 2>/dev/null || true
}

kill_remote_sessions() {
  local remote_host session ssh_target
  remote_host="$(server_cfg vllm.remote.host)"
  ssh_target="$(server_remote_ssh_target vllm)"
  session="$(server_cfg vllm.remote.session_name)"
  echo "Killing remote tmux session: $session for planner host $remote_host via SSH target $ssh_target"
  ssh "$ssh_target" "tmux kill-session -t $session 2>/dev/null || true"

  remote_host="$(server_cfg pi0.remote.host)"
  ssh_target="$(server_remote_ssh_target pi0)"
  session="$(server_cfg pi0.remote.session_name)"
  echo "Killing remote tmux session: $session for Pi0 host $remote_host via SSH target $ssh_target"
  ssh "$ssh_target" "tmux kill-session -t $session 2>/dev/null || true"
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
