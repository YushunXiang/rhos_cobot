#!/bin/bash

# 脚本：启动机器人ros环境
# 说明：此脚本将打开多个终端窗口来启动相机、查看图像、配置机械臂和准备数据采集环境。
# 警告：此脚本中包含了硬编码的sudo密码，请确保仅在受信任的安全环境中使用。

# 定义您的sudo密码
SUDO_PASSWORD="agx"

# 1. 新建终端，启动相机
gnome-terminal --title="Astra Camera" -- bash -c "echo '启动Astra多相机...'; roslaunch astra_camera multi_camera.launch; exec bash"

# 等待一段时间，确保相机节点完全启动
sleep 5

# 2. 新建终端，打开相机观测，重复三次
for i in {1..3}
do
  gnome-terminal --title="RQT Image View $i" -- bash -c "echo '打开第 $i 个相机观测窗口...'; rqt_image_view; exec bash"
  sleep 1
done

# 3. 新建终端，进入机械臂包目录并使用密码配置CAN
gnome-terminal --title="Piper Arm - Config" -- bash -c "\
echo '进入机械臂工作目录并配置CAN...'; \
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/; \
echo '正在使用提供的密码自动执行sudo命令...'; \
echo $SUDO_PASSWORD | sudo -S bash ./can_config.sh; \
echo 'CAN配置脚本执行完毕。'; \
exec bash"

# 等待CAN配置完成
sleep 3

# 4. 新建终端，启动主从机械臂（采集模式）
# gnome-terminal --title="Piper Arm - Start" -- bash -c "echo '启动主从机械臂（采集模式）...'; cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/; roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false; exec bash"

# 5. 新建终端，激活 Conda 环境并进入 rhos_cobot 代码目录
gnome-terminal --title="Data Collection - rhos_cobot" -- bash -c "echo '激活Conda环境并进入代码目录...'; source /home/agilex/miniconda3/etc/profile.d/conda.sh && conda activate aloha; cd /home/agilex/rhos_cobot; exec zsh"

echo "所有终端已按要求启动。"