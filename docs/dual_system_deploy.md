# VLM + VLA 双系统推理部署指南

本文档覆盖启用 `--use-llm-planner` 的场景：vLLM（qwen-vl）planner 服务 + pi0（OpenPI）policy 服务两层架构，支持任务拆解、底盘导航、离线回放等完整工作流。

- 纯手臂最小部署（不启用 LLM planner）请看 [`deploy.md`](deploy.md)。
- 离线回放调试（`run_piper_replay_mock.sh`）在本文档第 8 节；hybrid/planner replay 都是双系统路径，纯 policy replay 作为 §8 的一个子模式。
- 所有 `--use-llm-planner --use-robot-base` 运行都会在启动前要求人工 `yes` 确认。

## 0. 安全前提

在做 TRACER 2.0 deploy 推理前，先阅读本地手册：

- `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`

部署前至少确认以下手册约束：

- 仅在开阔、可观察区域内运行，操作员全程保持视线可见。
- TRACER 2.0 不具备自动避障、防跌落或生物接近预警能力，所有移动责任由操作员和上层控制逻辑承担。
- 两侧急停必须处于释放状态。
- 当前电池电压建议高于 22.5V；低于 22.5V 会报警，低于 21.5V 底盘会切断驱动和外部扩展供电。
- 默认防护等级为 IP22，避免雨雪、积水和超出手册环境范围的使用场景。

只要本次运行会触发任何底盘移动，即启用了以下条件：

- `--use-llm-planner` 且 `--use-robot-base` 且 `--prompt` 非空

`examples/piper_real/main.py` 都会在运行前要求操作员输入 `yes` 进行一次显式确认；未确认时，底盘保持零速度，手臂阶段也不会启动。

## 1. 硬件预检

环境准备（`init.sh` / `conda activate aloha` / `init_deploy` / `servers.toml`）请先完成，参见 [`deploy.md`](deploy.md) §1。

### 1.1 TRACER 底盘预检

运行 deploy 前，人工确认：

- 车体尾部 Q6 总电源已打开，电压表工作正常。
- 左右尾部急停均已释放。
- 若使用遥控器，遥控器已开机并处于可接管状态。
- 载荷、环境温度和防护等级满足手册要求。

### 1.2 CAN / ROS bring-up

本仓库只消费 `/odom_raw`、发布 `/cmd_vel`，并不直接完成 TRACER 的 CAN bring-up。运行前必须先启动一个底盘驱动层，把 TRACER 的底盘状态桥接到 ROS。

如果你使用官方 AgileX `tracer_ros` / `ugv_sdk` 方案，可按手册给出的典型步骤先验证底盘链路：

```bash
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000
candump can0
```

然后启动你自己的 TRACER ROS bridge。官方 `tracer_bringup tracer_robot_base.launch` 可以作为参考；无论使用哪套 bring-up，必须满足：

- ROS 中存在可读的 `/odom_raw`（`nav_msgs/Odometry`）。
- ROS 中存在可写的 `/cmd_vel`（`geometry_msgs/Twist`）。
- 下层驱动会持续把 `/cmd_vel` 转成 CAN 控制帧；TRACER 手册说明控制帧超时 500ms 后底盘会进入通讯保护并停车。

### 1.3 控制模式要求

手册要求 TRACER 进入指令控制模式后才会响应外部控制：

- 遥控器 `SWB` 最上方：指令控制模式。
- 遥控器 `SWB` 中间：遥控控制模式。
- 遥控器有更高优先级；若遥控器未切到指令模式，外部 `/cmd_vel` 或 CAN 控制可能不会生效。

如果你当前底盘桥接的是官方 CAN 栈，这一步是 deploy 前必须检查的前置条件。

## 2. 从 `start_servers.sh` 到 `main.py` 的脚本模板

`bash scripts/start_servers.sh [local|remote]` 只负责按 `config/servers.toml` 拉起两个服务：

- qwen-vl / vLLM planner server
- pi0 / OpenPI policy server

真正执行推理的入口仍然是 `python -m examples.piper_real.main`。运行前先区分两条连接链路：

- `--host` / `--port`：连接 pi0 policy server（WebSocket）
- `--planner.base-url`：连接 qwen-vl / vLLM planner（OpenAI-compatible HTTP）

`main.py` 现在默认会在启动前做一次 fail-fast server 预检：pi0 会检查 websocket 握手和 `reset`，planner 会检查 `/v1/models`。如果你明确想跳过这一步，可传 `--skip-server-checks`；超时时间可用 `--server-check-timeout-sec` 调整。

`config/servers.toml` 里的 `*.remote.host` 通常是 SSH 别名或远端登录名；传给 `main.py` 的地址应当是机器人工作站实际可达的 IP 或主机名，而不是机械照抄 SSH 别名。

下面给出几种常见脚本模板。

### 2.1 本机启动两个服务，然后执行完整推理

适用于 qwen-vl、pi0 和 `examples/piper_real/main.py` 都运行在同一台机器。

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
PLANNER_HOST=127.0.0.1
PLANNER_PORT="$(python3 scripts/_read_toml.py "$CONFIG" vllm.port)"
PLANNER_MODEL="$(python3 scripts/_read_toml.py "$CONFIG" vllm.local.served_model_name)"

bash scripts/start_servers.sh local

# 等 tmux 中两个模型都完成加载后，再执行 main.py
python -m examples.piper_real.main \
  --host "$PI0_HOST" \
  --port "$PI0_PORT" \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url "http://$PLANNER_HOST:$PLANNER_PORT/v1" \
  --planner.model "$PLANNER_MODEL"
```

### 2.2 远端启动两个服务，然后在机器人工作站执行完整推理

适用于 `bash scripts/start_servers.sh remote` 通过 SSH 到其他机器启动 qwen-vl 和 pi0，而 `main.py` 仍在当前机器人工作站执行。

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/rhos_cobot
conda activate aloha
init_deploy
source examples/piper_real/.venv/bin/activate

CONFIG=config/servers.toml

# 这里填写机器人工作站真正可达的服务地址，不一定等于 config 里的 SSH host 别名
PI0_HOST="${PI0_HOST:-192.168.3.101}"
PLANNER_HOST="${PLANNER_HOST:-192.168.3.123}"
PI0_PORT="$(python3 scripts/_read_toml.py "$CONFIG" pi0.port)"
PLANNER_PORT="$(python3 scripts/_read_toml.py "$CONFIG" vllm.port)"
PLANNER_MODEL="$(python3 scripts/_read_toml.py "$CONFIG" vllm.remote.served_model_name)"

bash scripts/start_servers.sh remote

# 等远端 tmux 中模型加载完成后，再执行 main.py
python -m examples.piper_real.main \
  --host "$PI0_HOST" \
  --port "$PI0_PORT" \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url "http://$PLANNER_HOST:$PLANNER_PORT/v1" \
  --planner.model "$PLANNER_MODEL"
```

### 2.3 只做导航，不连接策略服务

这种场景只需要 planner server，不需要 pi0 policy server，也不需要 `--host` / `--port`。

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/rhos_cobot
conda activate aloha
init_deploy
source examples/piper_real/.venv/bin/activate

CONFIG=config/servers.toml
PLANNER_HOST="${PLANNER_HOST:-192.168.3.123}"
PLANNER_PORT="$(python3 scripts/_read_toml.py "$CONFIG" vllm.port)"
PLANNER_MODEL="$(python3 scripts/_read_toml.py "$CONFIG" vllm.remote.served_model_name)"

bash scripts/start_vllm_server.sh

python -m examples.piper_real.main \
  --use-llm-planner \
  --use-robot-base \
  --navigation-only \
  --prompt "依次移动到厨房和客厅" \
  --planner.base-url "http://$PLANNER_HOST:$PLANNER_PORT/v1" \
  --planner.model "$PLANNER_MODEL"
```

## 3. CLI 开关

`examples/piper_real/main.py` 提供以下顶层开关：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use-llm-planner` | `False` | 启用两层架构：先拆解任务再按 subtask 执行 |
| `--use-robot-base` | `False` | navigate subtask 是否实际移动底盘（需要 `--use-llm-planner`） |
| `--navigation-only` | `False` | 只执行 navigate subtask，跳过 manipulate（需要 `--use-llm-planner`） |

Flag 组合规则：

- `--use-robot-base` 需要 `--use-llm-planner`
- `--navigation-only` 需要 `--use-llm-planner`
- `--replay-dataset` 与 `--use-llm-planner` 互斥
- `--replay-dataset` 与 `--navigation-only` 互斥

```bash
# LLM 拆解 + 导航移动 + 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8001 \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B

# LLM 拆解 + 导航仅打印（dry-run）+ 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8001 \
  --use-llm-planner \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B

# 仅导航（实际移动底盘，跳过操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --use-robot-base \
  --navigation-only \
  --prompt "依次移动到厨房和客厅" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B

# 仅导航 dry-run（仅打印计划，不移动不操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --navigation-only \
  --prompt "依次移动到厨房和客厅" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B
```

## 4. 底盘 ROS 话题

底盘相关的 ROS 话题定义在 `examples/piper_real/real_env.py` 的 `ros_config` 中：

| 话题 | 说明 |
|---|---|
| `/odom_raw` | 里程计（`nav_msgs/Odometry`），包含位置和姿态 |
| `/cmd_vel` | 速度指令（`geometry_msgs/Twist`），底盘接收的控制话题 |

底盘移动通过 `RosOperator.robot_base_publish([linear_x, angular_z])` 发布到 `/cmd_vel`。

## 5. LLM 两层任务架构（`--use-llm-planner`）

启用后，系统先调用 LLM 将完整 prompt 拆解为有序的 subtask 列表（navigate + manipulate），然后按序执行。

### 两层架构

1. **TaskDecomposer**: 一次性 LLM 调用，将 prompt 拆解为 subtask 列表。
2. **navigation_tool + Runtime**:
   - `navigate` subtask 调用 `examples/piper_real/navigation_tool.py`，基于 `/odom_raw` 反馈把底盘闭环驱动到一组固定 body-frame 坐标目标（语义与 `scripts/run_tracer_demo_sequence_3term.sh` 等价）。
   - `manipulate` subtask 继续走现有 OpenPI Runtime

### 执行流程

1. LLM 拆解 prompt 为 `{"subtasks": [{"type": "navigate"|"manipulate", "prompt": "..."}]}`。
2. 如果 `--use-robot-base` 且有 navigate subtask，要求操作员输入 `yes` 确认。
3. 按序执行每个 subtask：
   - **navigate**: 调用共享 navigation tool；`--use-robot-base` 时实际移动底盘，否则 dry-run。
   - **manipulate**: `--navigation-only` 时跳过；否则启动策略推理。
4. navigate 失败时终止整个任务，不执行后续 subtask。
5. 每个 manipulate subtask 独立运行一次策略推理。

### Planner 配置参数

任务拆解服务通过 `--planner.*` 传入：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--planner.base-url` | `http://192.168.3.123:8000/v1` | planner 服务地址 |
| `--planner.model` | `Qwen/Qwen3.5-4B` | 模型名称 |
| `--planner.api-key` | `EMPTY` | API 密钥 |

Planner 返回 JSON only，例如：

```json
{
  "subtasks": [
    {"type": "navigate", "prompt": "move to the table"},
    {"type": "manipulate", "prompt": "pick up the red cup"}
  ]
}
```

### Navigation tool 默认 routine

`navigation_tool.navigate()` 默认执行以下 5 个 body-frame 目标（相对 `navigate()` 调用时刻的起始位姿），与 `scripts/run_tracer_demo_sequence_3term.sh` 完全对齐：

| # | goal_x | goal_y | goal_yaw | 含义 |
|---|---|---|---|---|
| 1 | -0.3 | 0.0 | 0.0 | 后退 0.3 m |
| 2 | -0.3 | 0.0 | π/2 | 原地左转 90° |
| 3 | -0.3 | 0.6 | π/2 | 前进 0.6 m |
| 4 | -0.3 | 0.6 | 0.0 | 原地右转 90° |
| 5 | 0.0 | 0.6 | 0.0 | 前进 0.3 m |

每个目标通过 `navigate_to_goal` 以 `/odom_raw` 反馈闭环驱动；相邻目标之间 `sleep(1s)` 并补零速。任一目标失败立刻终止整条序列、补零速。`--use-robot-base` 未设置时 navigate 只打印每个目标（dry-run），不订阅 odom、不发布 `/cmd_vel`。

### 安全机制

- navigate subtask 开始前只做一次人工安全确认。
- 每个坐标目标到达后、以及两目标之间都会补发零速度到底盘。
- 若订阅不到 `/odom_raw`，navigation tool 会在 `odom_wait_timeout_s` 内 fail-fast，补零速并返回失败。
- 导航失败时终止整个任务，手臂操作不会启动。
- 运行退出时会再补发一次零速度到底盘。

## 6. 操作阶段底盘行为

操作（manipulate）阶段策略输出固定为 14 维（仅手臂关节），不包含底盘控制。底盘移动只发生在 navigate subtask 中，并由 `examples/piper_real/navigation_tool.py` 通过 `/odom_raw` 闭环控制执行。

如果策略模型输出超过 14 维，多余维度会被截断。

## 7. 验证

```bash
# 监控底盘速度指令
rostopic echo /cmd_vel

# 监控里程计
rostopic echo /odom_raw

# 如使用 CAN-USB，验证底盘总线回包
candump can0
```

检查项：

- 存在 navigate subtask 且启用 `--use-robot-base` 时，导航阶段先于手臂操作；每轮移动后有零速度停车。
- manipulate 阶段不发布 `/cmd_vel`；底盘移动只发生在 navigate subtask 中。
- 仅手臂操作时，`/cmd_vel` 无任何发布。
- task decomposition、navigation tool 调用、固定动作步骤和最终结果都能在日志中看到。
- 若动作缺少底盘维度、导航失败或未确认安全，`Runtime.run()` 不会继续或会在首次非法底盘动作处终止。

## 7.1 TRACER 底盘动作序列演示（`run_tracer_demo_sequence_3term.sh`）

`scripts/run_tracer_demo_sequence_3term.sh` 用于在不接入 LLM planner / pi0 的情况下，单独验证 TRACER 底盘按预设坐标依次移动的能力。脚本会用 `gnome-terminal` 一次性拉起三个终端：

- **Terminal A**：`roscore`。
- **Terminal B**：`roslaunch tracer_bringup tracer_robot_base.launch`，启动底盘 ROS bring-up。
- **Terminal C**：依次调用 `scripts/tracer/tracer_demo_coordinates.py`，订阅 `/odom`、发布 `/cmd_vel`，按 `--goal-x / --goal-y / --goal-yaw` 走完整段序列。

预期行为：从水池走到餐桌的固定动作序列，依次执行：

1. 后退 0.3 m（`goal_x=-0.3, goal_y=0.0, goal_yaw=0.0`）
2. 原地左转 90°（`goal_yaw=1.5708`）
3. 前进 0.6 m（`goal_x=0.6`）
4. 原地右转 90°（`goal_yaw=-1.5708`）
5. 前进 0.3 m（`goal_x=0.3`）

每步之间会 `sleep` 1 秒；完成后 Terminal C 输出 `✅ Sequence completed.`。

运行方式：

```bash
bash scripts/run_tracer_demo_sequence_3term.sh
```

运行前需满足 §1 中 TRACER 的硬件预检（急停释放、遥控器切到指令控制模式、电压充足等）；该脚本不会触发 §0 中的 `yes` 安全确认，请操作员在执行前自行确认现场安全并保持视线可见。

该 5 步序列与 `examples/piper_real/navigation_tool.navigate()` 的默认 routine 一致；部署路径下 `--use-llm-planner --use-robot-base` 的 navigate 子任务会复用同一控制律与 `/odom_raw` 反馈。

## 8. 离线回放调试（`--replay-dataset`）

使用已有 HDF5 数据集做 mock 推理验证：`main.py` 会读取回放观测，连接 pi0 policy server 做真实推理，但不需要实机、ROS 或底盘 bring-up。

推荐直接使用 `bash scripts/run_piper_replay_mock.sh ...`。该脚本现在默认只启动 pi0 policy server，不再为 replay mock 额外拉起 qwen-vl / vLLM planner。

### 8.1 推荐：直接用 replay mock 脚本

本机模式：

```bash
bash scripts/run_piper_replay_mock.sh local
```

远端模式：

```bash
PI0_HOST=192.168.3.101 bash scripts/run_piper_replay_mock.sh remote
```

如果你已经手动起好了 pi0，也可以直接跳过启动逻辑：

```bash
START_SERVERS=0 PYTHON_CMD=examples/piper_real/.venv/bin/python \
  bash scripts/run_piper_replay_mock.sh none
```

若你明确想把 qwen-vl / vLLM 也一起启动，再显式传：

```bash
START_TARGET=all bash scripts/run_piper_replay_mock.sh local
```

### 8.1.1 Replay 后端：policy / planner / hybrid

脚本支持三种 replay 后端，通过 `REPLAY_MODE` 环境变量或 wrapper flag 选择：

| 后端 | Flag | 环境变量 | 说明 |
|---|---|---|---|
| `policy`（默认） | `--policy-replay` | `REPLAY_MODE=policy` | 仅 pi0 policy replay |
| `planner` | `--planner-replay` | `REPLAY_MODE=planner` | 离线 VLM planner replay，不走 pi0 |
| `hybrid` | `--hybrid-replay` | `REPLAY_MODE=hybrid` | VLM 拆任务/重规划 + 本地 navigation tool + pi0 操作的混合 replay |

`hybrid` 模式需要 planner + pi0 两个服务，脚本会自动把 `START_TARGET=pi0` 提升为 `all`。其中 planner 仅负责 task decomposition 和 manipulate prompt replan；`navigate` 子任务在 replay 模式下直接跳过（只打印日志，不模拟底盘运动），因为离线数据集没有可用的 `/odom_raw` 反馈可驱动坐标闭环。相关环境变量：

- `PLANNER_HOST` / `PLANNER_PORT` / `PLANNER_MODEL`：planner 服务地址、端口、模型名；未设置时从 `config/servers.toml` 取。
- `MANIPULATE_MAX_STEPS`（默认 64）：hybrid 模式下每个 manipulate subtask 的策略步数上限。
- `MANIPULATE_REPLAN_INTERVAL_STEPS`（默认 16）：hybrid 模式下 VLM prompt 重规划的策略步间隔。
- `WAIT_FOR_PLANNER_READY` / `PLANNER_READY_TIMEOUT_SEC` / `PLANNER_READY_RETRY_INTERVAL_SEC` / `PLANNER_READY_CHECK_TIMEOUT_SEC`：planner 预检开关与超时。

```bash
REPLAY_MODE=hybrid START_TARGET=all bash scripts/run_piper_replay_mock.sh local
# 或：
bash scripts/run_piper_replay_mock.sh --hybrid-replay local
```

`planner` 模式仅依赖 vLLM planner server，不启 pi0，并保留离线逐步 VLM 导航调试能力。专属环境变量 `NAVIGATION_ONLY`（默认 `1`）：为 `1` 时过滤掉 decomposition 日志里的 manipulate 子任务。`START_TARGET` 合法值在此模式下为 `planner`（默认）或 `all`。等价调用（替代旧的 `run_piper_replay_planner.sh`）：

```bash
# 本机模式：脚本启动 vLLM planner
bash scripts/run_piper_replay_mock.sh --planner-replay local

# 远端模式
PLANNER_HOST=192.168.3.123 bash scripts/run_piper_replay_mock.sh --planner-replay remote

# 跳过 server 启动，连接到已运行的 planner
START_SERVERS=0 bash scripts/run_piper_replay_mock.sh --planner-replay none -- --skip-server-checks

# 同时启动 planner + pi0（通常不需要，planner 模式不用 pi0）
START_TARGET=all bash scripts/run_piper_replay_mock.sh --planner-replay local
```

### 8.1.2 `mock` 模式：无真实 pi0 时的一键 mock

当真实 pi0 policy server 不可用时，用 `mock` 模式让脚本自动启动内建 mock server（`examples.piper_real.mock_policy_server`）：

```bash
bash scripts/run_piper_replay_mock.sh mock
```

效果：

- 跳过 `start_pi0_server*.sh` / `start_servers.sh` 启动流程；
- 在后台启动 mock policy server 并绑定到 `$PI0_HOST:$PI0_PORT`（默认 `127.0.0.1:$PI0_PORT`，可通过环境变量覆盖）；
- 脚本退出时通过 EXIT trap 自动 kill mock 进程。

与其它 MODE 的关系：`mock` 是"pi0 来源"维度下的第 4 个取值，与 `local/remote/none` 对齐。

| MODE | pi0 server 由谁负责 |
|---|---|
| `local` | 脚本启动真实 pi0（本地） |
| `remote` | 脚本启动真实 pi0（远端） |
| `none` | 用户自己预先起好服务 |
| `mock` | 脚本自动起内建 mock，并在退出时清理 |

注意：

- `REPLAY_MODE=planner` 不使用 pi0，`mock` 对它无效，脚本会直接报错退出。
- `REPLAY_MODE=hybrid` + `mock`：只 mock pi0，planner 仍需用户自行准备。

旧的 `--mock-pi0` flag 保留为兼容别名，触发时会打印 deprecation 警告并切换到 `MODE=mock`。

### 8.1.3 可视化与 MP4 录制

- `--visualize`：replay 过程中显示相机画面和 subtask overlay。
- `--save-path PATH` 或 `SAVE_PATH=PATH`：把 replay 可视化保存为 MP4（转发到 `main.py --save-path`）。

```bash
bash scripts/run_piper_replay_mock.sh --visualize --save-path /tmp/replay.mp4 local
```

### 8.1.4 已存在 replay 进程的处理

- `--kill-existing-replay`（默认启用）：启动前 SIGTERM 掉已有的 replay mock 进程。
- `--no-kill-existing-replay`：保留已有进程不动。
- `--replay-kill-grace-sec SECONDS`（默认 5）：SIGTERM 到 SIGKILL 的等待时间。

### 8.2 兼容方式：手动启动后执行 `main.py`

下面示例假设你在本机启动两个服务。若使用 `remote` 模式，把 `PI0_HOST` 改成机器人工作站可达的实际地址即可。

```bash
CONFIG=config/servers.toml
PI0_HOST=127.0.0.1
PI0_PORT="$(python3 scripts/_read_toml.py "$CONFIG" pi0.port)"

bash scripts/start_servers.sh local

python -m examples.piper_real.main \
  --host "$PI0_HOST" \
  --port "$PI0_PORT" \
  --prompt "long-horizon replay mock validation" \
  --replay-dataset /inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/aloha-data/long-horizon-demo/episode_4.hdf5 \
  --max-episode-steps 0
```

如果只想启动 pi0 policy server，也可以把 `bash scripts/start_servers.sh local` 换成 `bash scripts/start_pi0_server_local.sh`。

### 8.3 行为与限制

回放 mock 模式下：

- 会连接 pi0 policy server，因此 `--host` 和 `--port` 必填且必须可达。
- 不初始化 ROS，不访问真实相机，也不会发布 `/cmd_vel` 或机械臂话题。
- 每一步会把 HDF5 里的 `qpos + images + prompt` 送进 policy server，并记录预测动作。
- 回放结束后会输出汇总，包括实际执行步数、`arm_mae`，以及在模型输出包含底盘维度时的 `base_mae`。
- `--use-llm-planner`、`--use-robot-base`、`--navigation-only` 都和 `--replay-dataset` 互斥。
- `episode_4.hdf5` 一共有 3750 steps；若沿用默认 `--max-episode-steps 1000`，会被提前截断。因此完整 long-horizon 验证请显式传 `--max-episode-steps 0`。
