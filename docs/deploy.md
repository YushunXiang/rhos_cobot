# Deploy Scripts for Rhos Cobot

## 0. 环境准备
在进行部署推理前，请确保机器已经运行过初始化脚本。
```bash
# 初始化脚本
sh scripts/init.sh
```
拔下两个主臂（即遥操作臂）的航插线，重启机械臂插排；运行以下命令进入deploy模式。
```bash
conda activate aloha
# 如果使用的是 zsh，可以使用 init_deploy 快速切换deploy模式。
init_deploy
# 或者手动运行以下脚本
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```


sudo ip addr add 10.42.0.3/24 dev enp3s0
sudo ip link set enp3s0 up
source examples/piper_real/.venv/bin/activate
source ~/cobot_magic/openpi/examples/piper_real/.venv/bin/activate
python -m examples.piper_real.main
