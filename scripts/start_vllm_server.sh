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
VLLM_HOST="$(server_cfg vllm.host)"
VLLM_PORT="$(server_cfg vllm.port)"

MAX_NUM_SEQS="$(server_cfg_optional vllm.remote.max_num_seqs)"
MAX_MODEL_LEN="$(server_cfg_optional vllm.remote.max_model_len)"
GPU_MEMORY_UTIL="$(server_cfg_optional vllm.remote.gpu_memory_utilization)"
TENSOR_PARALLEL="$(server_cfg_optional vllm.remote.tensor_parallel_size)"
SWAP_SPACE="$(server_cfg_optional vllm.remote.swap_space)"
OMP_NUM_THREADS="$(server_cfg_optional vllm.remote.omp_num_threads)"
TOOL_CALL_PARSER="$(server_cfg_optional vllm.remote.tool_call_parser)"
REASONING_PARSER="$(server_cfg_optional vllm.remote.reasoning_parser)"
SPECULATIVE_CONFIG="$(server_cfg_optional vllm.remote.speculative_config)"
ENABLE_AUTO_TOOL_CHOICE="$(server_normalize_bool "$(server_cfg_optional vllm.remote.enable_auto_tool_choice)")"
TRUST_REMOTE_CODE="$(server_normalize_bool "$(server_cfg_optional vllm.remote.trust_remote_code)")"

FLAGS=""
[[ -n "$MAX_NUM_SEQS" ]]       && FLAGS="$FLAGS --max-num-seqs $MAX_NUM_SEQS"
[[ -n "$MAX_MODEL_LEN" ]]      && FLAGS="$FLAGS --max-model-len $MAX_MODEL_LEN"
[[ -n "$GPU_MEMORY_UTIL" ]]    && FLAGS="$FLAGS --gpu-memory-utilization $GPU_MEMORY_UTIL"
[[ -n "$TENSOR_PARALLEL" ]]    && FLAGS="$FLAGS --tensor-parallel-size $TENSOR_PARALLEL"
[[ -n "$SWAP_SPACE" ]]         && FLAGS="$FLAGS --swap-space $SWAP_SPACE"
[[ -n "$TOOL_CALL_PARSER" ]]   && FLAGS="$FLAGS --tool-call-parser $TOOL_CALL_PARSER"
[[ -n "$REASONING_PARSER" ]]   && FLAGS="$FLAGS --reasoning-parser $REASONING_PARSER"
[[ -n "$SPECULATIVE_CONFIG" ]] && FLAGS="$FLAGS --speculative-config '$SPECULATIVE_CONFIG'"
[[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]] && FLAGS="$FLAGS --enable-auto-tool-choice"
[[ "$TRUST_REMOTE_CODE" == "1" ]]       && FLAGS="$FLAGS --trust-remote-code"

ENV_PREFIX=""
[[ -n "$OMP_NUM_THREADS" ]] && ENV_PREFIX="OMP_NUM_THREADS=$OMP_NUM_THREADS "

REMOTE_CMD="cd $WORK_DIR && ${ENV_PREFIX}uv run vllm serve $MODEL_PATH --served-model-name $SERVED_MODEL_NAME --host $VLLM_HOST --port $VLLM_PORT${FLAGS}"
ENCODED_CMD=$(printf '%s' "$REMOTE_CMD" | base64 -w 0)

echo "Stopping existing '$SESSION_NAME' session (if any) via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

echo "Starting vLLM planner server at http://$REMOTE_HOST:$VLLM_PORT/v1 via SSH target '$SSH_TARGET'..."
ssh "$SSH_TARGET" "tmux new-session -d -s $SESSION_NAME \
  /bin/bash -l -c 'echo $ENCODED_CMD | base64 -d | bash -l; exec bash'"

echo "Server starting in tmux session '$SESSION_NAME' for planner host '$REMOTE_HOST' via SSH target '$SSH_TARGET'."
echo "Expected planner base URL: http://$REMOTE_HOST:$VLLM_PORT/v1"
echo "Expected planner model: $SERVED_MODEL_NAME"
echo "To view logs:  ssh $SSH_TARGET -t 'tmux attach -t $SESSION_NAME'"
