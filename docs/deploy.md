# Deploy 部署指南

> 本文档只覆盖通用部署环境准备和纯手臂最小运行。
> 如需启用 VLM + VLA 双系统（`--use-llm-planner`、底盘导航、两层任务架构、离线 replay），请参见 [`dual_system_deploy.md`](dual_system_deploy.md)。

## 1. 环境准备与预检

```bash
cd ~/rhos_cobot
sh scripts/init.sh
```

拔下两个主臂（即遥操作臂）的航插线，重启机械臂插排；运行以下命令进入 deploy 模式：

```bash
conda activate aloha
# 如果使用的是 zsh，可以使用 init_deploy 快速切换 deploy 模式
init_deploy
```

激活 Python 环境：

```bash
source examples/piper_real/.venv/bin/activate
```

首次部署前先准备服务端配置：

```bash
cp config/servers.example.toml config/servers.toml
```

然后按你的实际环境修改 `config/servers.toml` 里的主机名、模型路径、checkpoint 路径和端口。该文件用于本地部署，不应提交到 git。

## 2. 只做操作，不启用 LLM planner

这种场景不需要 qwen-vl / vLLM，只需要 pi0 policy server。

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/rhos_cobot
conda activate aloha
init_deploy
source examples/piper_real/.venv/bin/activate

CONFIG=config/servers.toml
PI0_HOST=127.0.0.1
PI0_PORT="$(python3 scripts/_read_toml.py "$CONFIG" pi0.port)"

bash scripts/start_pi0_server_local.sh

python -m examples.piper_real.main \
  --host "$PI0_HOST" \
  --port "$PI0_PORT" \
  --prompt "turn on the water tap."
```

## 3. 最小运行命令

不使用 LLM planner，不使用底盘控制，仅手臂操作：

```bash
python -m examples.piper_real.main \
  --host 192.168.3.101 \
  --port 8001 \
  --prompt "turn on the water tap."
```

`--host` 和 `--port` 对应 pi0 policy server 的地址和端口；只有在启用 `--use-llm-planner` 时，才需要额外配置 `--planner.base-url` 指向 qwen-vl / vLLM planner——详见 [`dual_system_deploy.md`](dual_system_deploy.md)。
