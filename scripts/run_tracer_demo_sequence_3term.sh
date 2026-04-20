#!/usr/bin/env bash
set -euo pipefail

# ========= 配置 =========
VENV_ACTIVATE="/home/agilex/rhos_cobot-001-llm-navigation-stage/examples/piper_real/.venv/bin/activate"
PROJECT_ROOT="/home/agilex/rhos_cobot-001-llm-navigation-stage"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/tracer/tracer_demo_coordinates.py"

# 启动等待时间（秒）
WAIT_ROSCORE=3
WAIT_BASE=3
STEP_PAUSE=1

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
run_step '0.0' '0.0' '1.57079632679' 'Turn left 90 deg'
sleep '${STEP_PAUSE}'

# 3) 前进 0.6m
run_step '0.6' '0.0' '0.0' 'Forward 0.6m'
sleep '${STEP_PAUSE}'

# 4) 原地右转 90°
run_step '0.0' '0.0' '-1.57079632679' 'Turn right 90 deg'
sleep '${STEP_PAUSE}'

# 5) 前进 0.3m
run_step '0.3' '0.0' '0.0' 'Forward 0.3m'

echo '✅ Sequence completed.'
exec bash
"