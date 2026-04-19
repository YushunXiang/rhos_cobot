#!/usr/bin/env bash
# Start the configured remote Pi0 inference server in a tmux session.
# Usage: bash scripts/start_pi0_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

server_require_config

REMOTE_HOST="$(server_cfg pi0.remote.host)"
SSH_TARGET="$(server_remote_ssh_target pi0)"
SESSION_NAME="$(server_cfg pi0.remote.session_name)"
WORK_DIR="$(server_cfg pi0.remote.work_dir)"
POLICY_CONFIG="$(server_cfg_optional pi0.remote.policy_config)"
if [[ -z "$POLICY_CONFIG" ]]; then
  POLICY_CONFIG="$(server_cfg_optional pi0.remote.model)"
fi
if [[ -z "$POLICY_CONFIG" ]]; then
  POLICY_CONFIG="$(server_cfg pi0.policy_config)"
fi
CHECKPOINT="$(server_cfg pi0.remote.checkpoint)"
PI0_PORT="$(server_cfg pi0.port)"
REMOTE_OPENPI_ROOT="$WORK_DIR/third_party/openpi"
REMOTE_CHECKPOINT_PATH="$WORK_DIR/$CHECKPOINT"

echo "Stopping existing '$SESSION_NAME' session (if any) via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting Pi0 inference server at ws://$REMOTE_HOST:$PI0_PORT via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux new-session -d -s $SESSION_NAME \
  'export LD_LIBRARY_PATH=/home/Xtrainer/anaconda3/envs/brs/lib:\$LD_LIBRARY_PATH && \
   bash -l -c \"cd $REMOTE_OPENPI_ROOT && source .venv/bin/activate && uv run --active scripts/serve_policy.py --port=$PI0_PORT policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$REMOTE_CHECKPOINT_PATH\"; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' for Pi0 host '$REMOTE_HOST' via SSH target '$SSH_TARGET'."
echo "Expected Pi0 websocket: ws://$REMOTE_HOST:$PI0_PORT"
echo "To view logs:  ssh $SSH_TARGET -t 'tmux attach -t $SESSION_NAME'"
