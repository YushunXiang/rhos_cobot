#!/usr/bin/env bash
# Start the inference server on xtrainer-local in a tmux session.
# Usage: bash scripts/start_server.sh

set -euo pipefail

REMOTE_HOST="xtrainer-local"
SESSION_NAME="aloha-infer"
WORK_DIR="/mnt/sda/yushun/all-in-one-vla-inference"
MODEL="pi05_turn_on_tap"
CHECKPOINT="checkpoints/aloha-ckpt/pi05_turn_on_tap/pi05_turn_on_tap_20260323_145925/30000"

echo "Stopping existing '$SESSION_NAME' session (if any)..."
ssh "$REMOTE_HOST" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting inference server on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "tmux new-session -d -s $SESSION_NAME \
  'export LD_LIBRARY_PATH=/home/Xtrainer/anaconda3/envs/brs/lib:\$LD_LIBRARY_PATH && \
   bash -l -c \"cd $WORK_DIR && bash scripts/pi05_aloha_server.sh $MODEL $CHECKPOINT\"; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' on $REMOTE_HOST."
echo "To view logs:  ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
