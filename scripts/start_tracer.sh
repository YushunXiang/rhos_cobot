#!/bin/bash

echo "=== 准备启动 ROS 系统及驱动 ==="

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
VENV_ACTIVATE=${VENV_ACTIVATE:-"${PROJECT_ROOT}/examples/piper_real/.venv/bin/activate"}
TRACER_SCRIPT_DIR="${PROJECT_ROOT}/scripts/tracer"

# 1. 启动 roscore
echo ">>> [1/4] 启动 roscore..."
gnome-terminal --title="roscore" -- bash -c "roscore; exec bash"
sleep 3  # 暂停 3 秒，等待 roscore 完全启动

# 2. 启动 Piper 机械臂驱动节点
echo ">>> [2/4] 启动 Piper 机械臂驱动..."
gnome-terminal --title="piper_driver" -- bash -c "roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true; exec bash"
sleep 2

# 3. 启动 Tracer 2.0 移动底盘驱动
echo ">>> [3/4] 启动 Tracer 2.0 底盘驱动..."
gnome-terminal --title="tracer_driver" -- bash -c "cd ~/catkin_ws/src && roslaunch tracer_bringup tracer_robot_base.launch; exec bash"
sleep 4  # 暂停 4 秒，等待 CAN 通讯建立

# 4. 激活虚拟环境并运行 Python 控制脚本
echo ">>> [4/4] 启动 tracer_demo.py 控制脚本..."
gnome-terminal --title="tracer_demo" -- bash -c "
source '${VENV_ACTIVATE}' && 
cd '${TRACER_SCRIPT_DIR}' && 
python tracer_demo.py; 
exec bash"

echo "=== 所有节点已启动！==="
