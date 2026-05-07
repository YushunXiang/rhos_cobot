#!/usr/bin/env bash
set -euo pipefail

# ========= 配置 =========
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
VENV_ACTIVATE=${VENV_ACTIVATE:-"${PROJECT_ROOT}/examples/piper_real/.venv/bin/activate"}
SCRIPT_PATH="${PROJECT_ROOT}/scripts/tracer/tracer_demo_coordinates.py"

# 启动等待时间（秒）
WAIT_ROSCORE=3
WAIT_BASE=3
STEP_PAUSE=1

CAN_IFACE="can0"
CAN_BITRATE="500000"

# ========= can0 预检 =========
ensure_can_up() {
  if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
    echo "[3term] ERROR: ${CAN_IFACE} not found. Check USB-CAN adapter."
    exit 1
  fi
  if ip -details link show "${CAN_IFACE}" | grep -q "state UP"; then
    echo "[3term] ${CAN_IFACE} already UP."
    return 0
  fi
  echo "[3term] bringing up ${CAN_IFACE} @ ${CAN_BITRATE} bps (sudo required)..."
  sudo ip link set "${CAN_IFACE}" down 2>/dev/null || true
  sudo ip link set "${CAN_IFACE}" type can bitrate "${CAN_BITRATE}"
  sudo ip link set "${CAN_IFACE}" up
  if ! ip -details link show "${CAN_IFACE}" | grep -q "state UP"; then
    echo "[3term] ERROR: failed to bring ${CAN_IFACE} up."
    exit 1
  fi
  echo "[3term] ${CAN_IFACE} is UP."
}
ensure_can_up

# ========= 终端A：roscore =========
gnome-terminal --title="Terminal A - roscore" -- bash -c "
source '${VENV_ACTIVATE}'
echo '[Terminal A] starting roscore...'
roscore
exec bash
"

sleep "${WAIT_ROSCORE}"

# ========= 终端B：底盘驱动 =========
gnome-terminal --title="Terminal B - tracer_base" -- bash -c "
source '${VENV_ACTIVATE}'
echo '[Terminal B] launching tracer base...'
roslaunch tracer_bringup tracer_robot_base.launch
exec bash
"

sleep "${WAIT_BASE}"

# ========= 终端C：动作序列 =========
gnome-terminal --title="Terminal C - tracer sequence" -- bash -c "
set -euo pipefail
source '${VENV_ACTIVATE}'
cd '${PROJECT_ROOT}/scripts/tracer'
export PYTHONPATH='${PROJECT_ROOT}':\$PYTHONPATH

echo '[Terminal C] waiting for /odom data (timeout 10s)...'
if ! timeout 10 rostopic echo -n1 /odom >/dev/null 2>&1; then
  echo '[Terminal C] ERROR: no data on /odom. Check chassis power / e-stop / CAN wiring.'
  exec bash
fi
echo '[Terminal C] /odom OK.'

run_step() {
  local gx=\"\$1\"
  local gy=\"\$2\"
  local gyaw=\"\$3\"
  local desc=\"\$4\"

  echo '=================================================='
  echo \"[STEP] \${desc}\"
  echo \"goal_x=\${gx}, goal_y=\${gy}, goal_yaw=\${gyaw}\"
  echo '=================================================='

  python '${SCRIPT_PATH}' \
    --goal-x \"\${gx}\" \
    --goal-y \"\${gy}\" \
    --goal-yaw \"\${gyaw}\" \
    --odom-topic /odom \
    --cmd-vel-topic /cmd_vel
}

echo '[Terminal C] start sequential motion...'

# 1) 后退 0.3m
run_step '-0.3' '0.0' '0.0' 'Backward 0.3m'
sleep '${STEP_PAUSE}'

# 2) 原地左转 90°
run_step '-0.3' '0.0' '1.57079632679' 'Turn left 90 deg'
sleep '${STEP_PAUSE}'

# 3) 前进 0.6m
run_step '-0.3' '0.6' '1.57079632679' 'Forward 0.6m'
sleep '${STEP_PAUSE}'

# 4) 原地右转 90°
run_step '-0.3' '0.6' '0.0' 'Turn right 90 deg'
sleep '${STEP_PAUSE}'

# 5) 前进 0.3m
run_step '0.0' '0.6' '0.0' 'Forward 0.3m'

echo '✅ Sequence completed.'
exec bash
"
