#!/usr/bin/env bash
# Start the configured remote vLLM planner server in a tmux session.
# Usage: bash scripts/start_vllm_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

server_require_config

REMOTE_HOST="$(server_cfg vllm.remote.host)"
SSH_TARGET="$(server_remote_ssh_target vllm)"
SESSION_NAME="$(server_cfg vllm.remote.session_name)"
WORK_DIR="$(server_cfg vllm.remote.work_dir)"
MODEL_PATH="$(server_cfg vllm.remote.model_path)"
SERVED_MODEL_NAME="$(server_cfg vllm.remote.served_model_name)"
VLLM_PORT="$(server_cfg vllm.port)"

echo "Stopping existing '$SESSION_NAME' session (if any) via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting vLLM planner server at http://$REMOTE_HOST:$VLLM_PORT/v1 via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux new-session -d -s $SESSION_NAME \
  'bash -l -c \"cd $WORK_DIR && uv run vllm serve $MODEL_PATH --served-model-name $SERVED_MODEL_NAME --port $VLLM_PORT\"; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' for planner host '$REMOTE_HOST' via SSH target '$SSH_TARGET'."
echo "Expected planner base URL: http://$REMOTE_HOST:$VLLM_PORT/v1"
echo "Expected planner model: $SERVED_MODEL_NAME"
echo "To view logs:  ssh $SSH_TARGET -t 'tmux attach -t $SESSION_NAME'"
