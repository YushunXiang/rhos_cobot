#!/usr/bin/env bash
set -euo pipefail

# =========================
# 用法:
#   bash run_tracer_coordinate_test.sh [GOAL_X] [GOAL_Y] [GOAL_YAW]
#
# 例子:
#   bash run_tracer_coordinate_test.sh              # 用默认目标
#   bash run_tracer_coordinate_test.sh 0.5 0.0      # 只改 x/y，yaw 用默认
#   bash run_tracer_coordinate_test.sh 1.2 -0.3 1.57
# =========================

# ---- 默认参数（可改）----
GOAL_X="${1:-0.5}"
GOAL_Y="${2:-0.0}"
GOAL_YAW="${3:-0.0}"

ODOM_TOPIC="/odom_raw"
CMD_VEL_TOPIC="/cmd_vel"

# 实机建议先保守速度
MAX_LINEAR="0.15"
MAX_ANGULAR="0.30"

PROJECT_ROOT="/home/agilex/rhos_cobot-001-llm-navigation-stage"
VENV_PATH="${PROJECT_ROOT}/.venv/bin/activate"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/tracer/tracer_demo_coordinates.py"

echo "=== TRACER coordinate test launcher ==="
echo "GOAL_X=${GOAL_X}, GOAL_Y=${GOAL_Y}, GOAL_YAW=${GOAL_YAW}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"

# ---- 前置检查 ----
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] 找不到脚本: ${SCRIPT_PATH}"
  exit 1
fi

if [[ ! -f "${VENV_PATH}" ]]; then
  echo "[ERROR] 找不到虚拟环境激活脚本: ${VENV_PATH}"
  exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "[ERROR] 未找到 gnome-terminal，请先安装或改为你自己的终端启动方式。"
  exit 1
fi

echo ">>> [1/4] 启动 roscore..."
gnome-terminal --title="roscore" -- bash -lc "roscore; exec bash"
sleep 3

echo ">>> [2/4] 启动 Tracer 底盘驱动..."
gnome-terminal --title="tracer_driver" -- bash -lc "roslaunch tracer_bringup tracer_robot_base.launch; exec bash"
sleep 5

echo ">>> [3/4] (可选) 启动 Piper 机械臂驱动..."
gnome-terminal --title="piper_driver" -- bash -lc "roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true; exec bash"
sleep 3

echo ">>> [4/4] 启动坐标导航脚本..."
gnome-terminal --title="tracer_demo_coordinates" -- bash -lc "
source '${VENV_PATH}' && \
cd '${PROJECT_ROOT}' && \
python '${SCRIPT_PATH}' \
  --goal-x '${GOAL_X}' \
  --goal-y '${GOAL_Y}' \
  --goal-yaw '${GOAL_YAW}' \
  --odom-topic '${ODOM_TOPIC}' \
  --cmd-vel-topic '${CMD_VEL_TOPIC}' \
  --max-linear-vel-mps '${MAX_LINEAR}' \
  --max-angular-vel-rad-s '${MAX_ANGULAR}'; \
exec bash
"

echo "=== 已发起全部终端。请先确认场地安全、可急停。==="