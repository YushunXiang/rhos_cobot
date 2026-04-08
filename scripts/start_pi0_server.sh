#!/usr/bin/env bash
# Start the inference server on xtrainer-local in a tmux session.
# Usage: bash scripts/start_pi0_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/config/servers.toml"

_cfg() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1"; }
_cfg_optional() { python3 "$SCRIPT_DIR/_read_toml.py" "$CONFIG" "$1" 2>/dev/null || true; }

REMOTE_HOST="$(_cfg pi0.remote.host)"
SESSION_NAME="$(_cfg pi0.remote.session_name)"
WORK_DIR="$(_cfg pi0.remote.work_dir)"
POLICY_CONFIG="$(_cfg_optional pi0.remote.policy_config)"
if [[ -z "$POLICY_CONFIG" ]]; then
  POLICY_CONFIG="$(_cfg_optional pi0.remote.model)"
fi
if [[ -z "$POLICY_CONFIG" ]]; then
  POLICY_CONFIG="$(_cfg pi0.policy_config)"
fi
CHECKPOINT="$(_cfg pi0.remote.checkpoint)"

echo "Stopping existing '$SESSION_NAME' session (if any)..."
ssh "$REMOTE_HOST" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting inference server on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "tmux new-session -d -s $SESSION_NAME \
  'export LD_LIBRARY_PATH=/home/Xtrainer/anaconda3/envs/brs/lib:\$LD_LIBRARY_PATH && \
   bash -l -c \"cd $WORK_DIR && bash scripts/pi05_aloha_server.sh $POLICY_CONFIG $CHECKPOINT\"; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' on $REMOTE_HOST."
echo "To view logs:  ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
