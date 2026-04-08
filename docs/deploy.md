# Deploy 部署指南

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

## 2. 最小运行命令

不使用 LLM planner，不使用底盘控制，仅手臂操作：

```bash
python -m examples.piper_real.main \
  --host 192.168.3.101 \
  --port 8000 \
  --prompt "turn on the water tap."
```

`--host` 和 `--port` 对应 xtrainer 的 IP 和 pi remote server 的端口，运行前请确认。

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
# 仅手臂操作（默认行为，不使用 LLM planner）
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --prompt "turn on the water tap."

# LLM 拆解 + 导航移动 + 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子"

# LLM 拆解 + 导航仅打印（dry-run）+ 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --prompt "移动到桌子旁边拿起红色杯子"

# 仅导航（实际移动底盘，跳过操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --use-robot-base \
  --navigation-only \
  --prompt "依次移动到厨房和客厅"

# 仅导航 dry-run（仅打印计划，不移动不操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --navigation-only \
  --prompt "依次移动到厨房和客厅"
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
2. **LLMNavigationPlanner**: 执行单个 navigate subtask 的多步导航循环。

### 执行流程

1. LLM 拆解 prompt 为 `[{type: "navigate"|"manipulate", prompt: "..."}]`。
2. 如果 `--use-robot-base` 且有 navigate subtask，要求操作员输入 `yes` 确认。
3. 按序执行每个 subtask：
   - **navigate**: `--use-robot-base` 时实际移动底盘；否则仅打印。
   - **manipulate**: `--navigation-only` 时跳过；否则启动策略推理。
4. navigate 失败时终止整个任务，不执行后续 subtask。
5. 每个 manipulate subtask 独立运行一次策略推理。

### Planner 配置参数

导航相关配置通过 `--planner.*` 传入（与之前相同）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--planner.base-url` | `http://192.168.3.123:8000/v1` | planner 服务地址 |
| `--planner.model` | `Qwen/Qwen3.5-4B` | 模型名称 |
| `--planner.api-key` | `EMPTY` | API 密钥 |
| `--planner.max-nav-steps` | `20` | 最大有效导航步数 |
| `--planner.max-linear-vel` | `0.3` | 底盘线速度上限 (m/s) |
| `--planner.max-angular-vel` | `0.5` | 底盘角速度上限 (rad/s) |
| `--planner.default-duration` | `1.5` | 单步默认执行时长 (s) |

### 安全机制

- 超速或非法 planner 指令会被拒绝并要求 planner 重新决策。
- planner 连续失败 3 次后重试，第 4 次失败终止导航。
- 导航失败时终止整个任务，手臂操作不会启动。
- 运行退出时会补发一次零速度到底盘。

## 6. 操作阶段底盘行为

操作（manipulate）阶段策略输出固定为 14 维（仅手臂关节），不包含底盘控制。底盘移动仅在 navigate subtask 中由 LLMNavigationPlanner 执行。

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

- `--use-llm-planner` 时，导航阶段先于手臂操作；每轮移动后有零速度停车。
- `--use-robot-base` 时，策略推理中 `/cmd_vel` 有速度输出，但不会超过配置上限。
- 两者均关闭时，`/cmd_vel` 无任何发布。
- planner 原始响应、拒绝/重试情况、执行速度和最终原因都能在日志中看到。
- 若动作缺少底盘维度、导航失败或未确认安全，`Runtime.run()` 不会继续或会在首次非法底盘动作处终止。

## 8. 离线回放调试（`--replay-dataset`）

使用已有 HDF5 数据集，逐步打印 ground-truth action，不需要推理服务器、实机或 ROS。

```bash
python -m examples.piper_real.main \
  --replay-dataset /home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5
```

回放模式下：

- 不连接推理服务器，不需要 `--host` 和 `--port`。
- 从 HDF5 文件逐帧打印 action（arm 14 维 + base 2 维，如有）。
- 回放结束后输出汇总（总步数、action 维度、是否包含 base_action）。
- `--use-llm-planner` 和 `--replay-dataset` 互斥。
