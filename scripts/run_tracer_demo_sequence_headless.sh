#!/usr/bin/env bash
set -euo pipefail

# ========= 配置 =========
VENV_ACTIVATE="/home/agilex/rhos_cobot-001-llm-navigation-stage/examples/piper_real/.venv/bin/activate"
PROJECT_ROOT="/home/agilex/rhos_cobot-001-llm-navigation-stage"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/tracer/tracer_demo_coordinates.py"

LOG_DIR="${PROJECT_ROOT}/logs/tracer_demo"
mkdir -p "${LOG_DIR}"
ROSCORE_LOG="${LOG_DIR}/roscore.log"
BASE_LOG="${LOG_DIR}/tracer_base.log"
SEQ_LOG="${LOG_DIR}/sequence.log"

WAIT_ROSCORE=3
WAIT_BASE=3
STEP_PAUSE=1

CAN_IFACE="can0"
CAN_BITRATE="500000"

# ========= can0 预检 =========
ensure_can_up() {
  if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
    echo "[headless] ERROR: ${CAN_IFACE} not found. Check USB-CAN adapter."
    exit 1
  fi
  if ip -details link show "${CAN_IFACE}" | grep -q "state UP"; then
    echo "[headless] ${CAN_IFACE} already UP."
    return 0
  fi
  echo "[headless] bringing up ${CAN_IFACE} @ ${CAN_BITRATE} bps (sudo required)..."
  sudo ip link set "${CAN_IFACE}" down 2>/dev/null || true
  sudo ip link set "${CAN_IFACE}" type can bitrate "${CAN_BITRATE}"
  sudo ip link set "${CAN_IFACE}" up
  if ! ip -details link show "${CAN_IFACE}" | grep -q "state UP"; then
    echo "[headless] ERROR: failed to bring ${CAN_IFACE} up."
    exit 1
  fi
  echo "[headless] ${CAN_IFACE} is UP."
}
ensure_can_up

# ========= 清理函数 =========
PIDS=()
cleanup() {
  echo "[headless] cleaning up background processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

# ========= A: roscore =========
echo "[headless] starting roscore (log: ${ROSCORE_LOG})..."
( roscore ) >"${ROSCORE_LOG}" 2>&1 &
PIDS+=($!)
sleep "${WAIT_ROSCORE}"

# ========= B: tracer base =========
echo "[headless] launching tracer base (log: ${BASE_LOG})..."
( roslaunch tracer_bringup tracer_robot_base.launch ) >"${BASE_LOG}" 2>&1 &
PIDS+=($!)
sleep "${WAIT_BASE}"

# ========= /odom 预检 =========
echo "[headless] waiting for /odom data (timeout 10s)..."
if ! timeout 10 rostopic echo -n1 /odom >/dev/null 2>&1; then
  echo "[headless] ERROR: no data on /odom. Check chassis power / e-stop / CAN wiring."
  exit 1
fi
echo "[headless] /odom OK."

# ========= C: 动作序列 =========
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}/scripts/tracer"

run_step() {
  local gx="$1" gy="$2" gyaw="$3" desc="$4"
  echo "=================================================="
  echo "[STEP] ${desc}"
  echo "goal_x=${gx}, goal_y=${gy}, goal_yaw=${gyaw}"
  echo "=================================================="
  python "${SCRIPT_PATH}" \
    --goal-x "${gx}" \
    --goal-y "${gy}" \
    --goal-yaw "${gyaw}" \
    --odom-topic /odom \
    --cmd-vel-topic /cmd_vel
}

{
  echo "[headless] start sequential motion..."
  run_step '-0.3' '0.0' '0.0' 'Backward 0.3m'; sleep "${STEP_PAUSE}"
  run_step '-0.3' '0.0' '1.57079632679' 'Turn left 90 deg'; sleep "${STEP_PAUSE}"
  run_step '-0.3' '0.6' '1.57079632679' 'Forward 0.6m'; sleep "${STEP_PAUSE}"
  run_step '-0.3' '0.6' '0.0' 'Turn right 90 deg'; sleep "${STEP_PAUSE}"
  run_step '0.0' '0.6' '0.0' 'Forward 0.3m'
  echo '[headless] Sequence completed.'
} 2>&1 | tee "${SEQ_LOG}"
