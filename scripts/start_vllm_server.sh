#!/usr/bin/env bash
# Start the remote vLLM planner server in a tmux session.
# Usage: bash scripts/start_vllm_server.sh

set -euo pipefail

REMOTE_HOST="nas-local"
SESSION_NAME="vllm-qwen-planner"
WORK_DIR="/home/web/yushun/llm-planner"
MODEL_PATH="models/Qwen/Qwen3.5-4B"
SERVED_MODEL_NAME="Qwen/Qwen3.5-4B"

echo "Stopping existing '$SESSION_NAME' session (if any)..."
ssh "$REMOTE_HOST" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting vLLM planner server on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "tmux new-session -d -s $SESSION_NAME \
  'bash -l -c \"cd $WORK_DIR && uv run vllm serve $MODEL_PATH --served-model-name $SERVED_MODEL_NAME\"; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' on $REMOTE_HOST."
echo "Expected planner base URL: http://192.168.3.123:8000/v1"
echo "Expected planner model: $SERVED_MODEL_NAME"
echo "To view logs:  ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
