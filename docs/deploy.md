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

只要本次运行会触发任何底盘移动，即启用了以下任一条件：

- `--use-llm-planner` 且 `--prompt` 非空
- `--use-robot-base`

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

`examples/piper_real/main.py` 提供两个独立的顶层开关，默认均为关闭：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use-llm-planner` | `False` | 是否在操作前启用 LLM 导航阶段 |
| `--use-robot-base` | `False` | 策略推理时是否将 action 中的底盘速度发布到 `/cmd_vel` |

两个开关互相独立，可以自由组合：

```bash
# 仅手臂操作（默认行为）
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --prompt "turn on the water tap."

# 先 LLM 导航到目标位置，再手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --prompt "移动到桌子旁边拿起红色杯子"

# 策略推理时同时控制底盘（策略输出包含底盘速度）
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-robot-base \
  --prompt "turn on the water tap."

# 先导航再执行，执行阶段也控制底盘
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子"
```

只要本次运行存在任何底盘运动，程序都会先要求一次人工确认。

## 4. 底盘 ROS 话题

底盘相关的 ROS 话题定义在 `examples/piper_real/real_env.py` 的 `ros_config` 中：

| 话题 | 说明 |
|---|---|
| `/odom_raw` | 里程计（`nav_msgs/Odometry`），包含位置和姿态 |
| `/cmd_vel` | 速度指令（`geometry_msgs/Twist`），底盘接收的控制话题 |

底盘移动通过 `RosOperator.robot_base_publish([linear_x, angular_z])` 发布到 `/cmd_vel`。

## 5. LLM 导航阶段（`--use-llm-planner`）

启用后，在手臂操作前插入一个 LLM 驱动的导航循环。流程：

1. 若 `--prompt` 非空，先显示底盘运动安全警告并要求操作员输入 `yes`。
2. 每轮循环读取前置相机图像和里程计，发送给 planner 服务。
3. planner 返回 `move` 或 `stop` 指令。
4. `move` 指令执行后立即停车，进入下一轮。
5. `stop` 指令表示到达目标位置，导航结束，进入手臂操作。

### Planner 服务要求

在机器人工作站可访问的位置启动一个 OpenAI 兼容的多模态服务，需满足：

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

### Planner 配置参数

导航相关配置通过 `--planner.*` 传入：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--planner.base-url` | `http://192.168.3.123:8000/v1` | planner 服务地址 |
| `--planner.model` | `Qwen/Qwen3.5-4B` | 模型名称 |
| `--planner.api-key` | `EMPTY` | API 密钥 |
| `--planner.max-nav-steps` | `20` | 最大有效导航步数 |
| `--planner.max-linear-vel` | `0.3` | 共享底盘线速度上限 (m/s)，不能超过手册上限 `1.8` |
| `--planner.max-angular-vel` | `0.5` | 共享底盘角速度上限 (rad/s)，不能超过手册上限 `1.0` |
| `--planner.default-duration` | `1.5` | 单步默认执行时长 (s) |

### 安全机制

- 超速或非法 planner 指令会被拒绝并要求 planner 重新决策。
- planner 连续失败 3 次后重试，第 4 次失败终止导航。
- 失败次数不消耗步数预算。
- 导航失败或未确认安全时，手臂操作不会启动。
- 运行退出时会补发一次零速度到底盘。

## 6. 策略推理中的底盘控制（`--use-robot-base`）

启用后，`PiperRealEnv.step()` 会从策略输出的 action 向量中提取底盘速度分量（`action[14:16]`，即 `[linear_x, angular_z]`），并发布到 `/cmd_vel`。

这适用于策略本身经过底盘移动训练的场景（如 mobile manipulation 数据集训练的模型）。未启用时，`step()` 仅发布手臂关节指令。

策略驱动底盘时额外约束如下：

- 仍然需要开跑前的人机安全确认。
- `action[14:16]` 必须存在，否则本次运行会拒绝该动作并停车。
- 速度同样受 `--planner.max-linear-vel` 和 `--planner.max-angular-vel` 约束。
- 若策略输出超限或非法数值，程序会先发零速度，再报错终止，避免无界底盘控制绕过安全门。

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

使用已有 HDF5 数据集作为观测输入，仅运行推理服务器，不需要实机、ROS 或底盘。适用于：

- 调试推理流水线和策略输出
- 对比预测动作与数据集中的 ground-truth 动作
- 验证模型在已知场景上的表现

```bash
# 使用 turn_on_off_tap 数据集回放
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --replay-dataset /home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5 \
  --prompt "turn on the water tap."
```

回放模式下：

- 跳过 ROS 初始化、安全确认和 LLM 导航阶段。
- 从 HDF5 文件逐帧读取观测（qpos + 3 路 RGB 图像），发送给推理服务器。
- 推理服务器返回的预测动作仅记录，不发布到机械臂或底盘。
- 回放结束后输出预测动作与 ground-truth 的 MAE（Mean Absolute Error）。
- `--use-llm-planner` 和 `--use-robot-base` 在回放模式下被忽略。
