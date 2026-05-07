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

## 4. Real deploy：hybrid + qz planner

`scripts/run_piper_deploy.sh` 不负责启动 pi0 policy server；它连接已经运行的 pi0，并在 `MODE=hybrid` 下调用 VLM planner 做导航/操作子任务调度。启动 checkpoint 时使用的 `POLICY_CONFIG`、`CHECKPOINT_DIR`、`PROGRESS_SOURCE` 需要传给 `scripts/start_pi0_server_local.sh` 或远端启动脚本，而不是依赖 `run_piper_deploy.sh` 自动拉起。

### 4.1 带 progress checkpoint

先启动带 progress head 的 pi0 server。若 pi0 在本机：

```bash
POLICY_CONFIG=pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3 \
CHECKPOINT_DIR=/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/openpi/checkpoints/pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3/pi05_pick_bread_leaf_progress_dual_20260424_061145/99999 \
PROGRESS_SOURCE=subtask \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_pi0_server_local.sh
```

然后运行 real deploy：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u socks_proxy -u SOCKS_PROXY \
    -u no_proxy -u NO_PROXY \
    MODE=hybrid \
    TASK_NAME=pick_bread_leaf_3 \
    TASK_SPEC=none \
    MANIPULATE_MAX_STEPS=10000 \
    MANIPULATE_REPLAN_INTERVAL_STEPS=100 \
    PROGRESS_HEAD_MODE=auto \
    PLANNER_BACKEND=qz \
    QZ_STATE_FILE=config/vllm_server_state.json \
    QZ_USE_PROXY=0 \
    bash scripts/run_piper_deploy.sh
```

说明：

- `PROGRESS_SOURCE=subtask` 让 pi0 server 返回子任务进度，适合 hybrid 的单个 manipulate subtask。
- `PROGRESS_HEAD_MODE=auto` 让 deploy 端根据 pi0 metadata 自动启用 progress-first 完成检测。
- 如果 pi0 不在本机，先用远端方式启动同一个 checkpoint，并确认 `PI0_HOST` 指向可达的 websocket 地址。

### 4.2 不带 progress checkpoint

先启动不带 progress head 的 pi0 server：

```bash
POLICY_CONFIG=pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3 \
CHECKPOINT_DIR=/path/to/non_progress_checkpoint \
PROGRESS_SOURCE=task \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_pi0_server_local.sh
```

然后关闭 deploy 端 progress-first 逻辑：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u socks_proxy -u SOCKS_PROXY \
    -u no_proxy -u NO_PROXY \
    MODE=hybrid \
    TASK_NAME=pick_bread_leaf_3 \
    TASK_SPEC=none \
    MANIPULATE_MAX_STEPS=10000 \
    MANIPULATE_REPLAN_INTERVAL_STEPS=100 \
    PROGRESS_HEAD_MODE=off \
    PLANNER_BACKEND=qz \
    QZ_STATE_FILE=config/vllm_server_state.json \
    QZ_USE_PROXY=0 \
    bash scripts/run_piper_deploy.sh
```

其中 `/path/to/non_progress_checkpoint` 需要替换为不带 progress head 的真实 checkpoint 目录。
