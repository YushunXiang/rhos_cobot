# Deploy Scripts for Rhos Cobot

## 0. 安全前提

在做 TRACER 2.0 deploy 推理前，先阅读本地手册：

- `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`

重点确认以下约束：

- 在开阔、可观察区域内运行。
- 操作员全程保持视线可见并可随时急停。
- 不要把底盘当成具备自动避障能力的平台。

`examples/piper_real/main.py` 在启用导航时会强制要求操作员显式确认，未确认不会开始底盘移动，也不会进入手臂操作阶段。

## 1. 环境准备

```bash
sh scripts/init.sh
```

如果需要，先配置机器人工作站网络：

```bash
sudo ip addr add 10.42.0.3/24 dev enp3s0
sudo ip link set enp3s0 up
```

准备 Python 环境：

```bash
uv venv --python 3.11 examples/piper_real/.venv
source examples/piper_real/.venv/bin/activate
uv pip compile examples/piper_real/requirements.in -o examples/piper_real/requirements.txt --python-version 3.11
uv pip sync examples/piper_real/requirements.txt
```

说明：当前仓库快照里没有 `packages/openpi-client`，因此请使用已经提供 `openpi_client` 的现有 deploy 环境。

## 2. 切换到底盘 deploy 模式

拔下两个主臂（遥操作臂）的航插线，重启机械臂插排后执行：

```bash
conda activate aloha
init_deploy
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
source examples/piper_real/.venv/bin/activate
```

## 3. 启动本地 planner 服务

在机器人工作站可访问的位置启动一个 OpenAI 兼容的多模态 planner 服务。它至少需要满足：

- 提供 chat-completions 接口。
- 支持图像 + 文本输入。
- 只返回 JSON planner 决策。

planner 返回格式必须是以下两种之一：

```json
{"action": "move", "linear_x": 0.2, "angular_z": -0.1, "duration": 1.2, "reasoning": "Rotate slightly and move forward."}
```

```json
{"action": "stop", "reason": "The robot is in a usable operating position."}
```

## 4. 先导航后操作

```bash
python -m examples.piper_real.main \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://localhost:8000/v1 \
  --planner.model qwen2.5-vl-72b
```

运行时流程：

1. 初始化 `PiperRealEnvironment`。
2. 若启用导航且 `--prompt` 非空，先显示安全警告并要求输入 `yes` 确认。
3. planner 循环读取前置相机与里程计，发布有界 `cmd_vel`，每轮动作后立即停车。
4. 只有在 planner 返回 `stop` 后，才启动 OpenPI 手臂策略。

默认速度限制来自 `PlannerConfig`：

- `linear_x`: `±0.3 m/s`
- `angular_z`: `±0.5 rad/s`
- `max_nav_steps`: `20`
- 连续 planner 失败重试：最多 `3` 次，第四次失败终止导航

## 5. 跳过导航直接操作

```bash
python -m examples.piper_real.main \
  --prompt "拿起红色杯子" \
  --planner.enable-navigation false
```

此路径会输出 `navigation skipped` 状态，不发送预操作阶段的底盘移动指令，随后直接进入手臂操作。

## 6. 验证方式

```bash
rostopic echo /cmd_vel
```

至少检查以下项目：

- 导航阶段确实先于手臂操作开始。
- planner 等待期间底盘保持停止。
- 每轮移动后都会发送零速度停车。
- planner 原始响应、拒绝/重试情况、执行速度和最终原因都能在日志中看到。
- 若导航失败或未确认安全，`Runtime.run()` 不会启动。
