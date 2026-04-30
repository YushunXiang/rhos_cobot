# Replay Scripts

## 0. 注意事项
执行这一节前，请先关闭所有终端，避免旧的 ROS 节点残留。

重播前请按以下顺序操作：

1. 将机器臂断电重启。
2. 先拔掉主臂（遥操作臂）的航插头，再启动从臂。
3. 确认机械臂周围无障碍物，确保 replay 过程中有足够安全空间。

本节使用手动启动方式，不建议在 replay 前直接执行 `scripts/init.sh`。

## 1. 启动 ROS 与从臂
终端 A：

```bash
roscore
```

终端 B：

```bash
conda activate aloha
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

其中 `mode:=1` 为 deploy 模式，允许从臂接收 replay 发布的关节指令。

## 2. 重播数据集
终端 C：

```bash
conda activate aloha
cd /home/agilex/rhos_cobot
python -m scripts.post_collect.replay_data --dataset_dir ~/data --task_name aloha_mobile_dummy --episode_idx 0
```

默认会发布以下消息：

- 彩色图像 topic
- 主臂关节 action topic
- 从臂关节 qpos topic
- 底盘速度 topic（如果数据集中存在 `base_action`）

发布数据包后，下游节点可订阅这些消息并按数据集进行重放。

## 3. 仅发布主臂关节消息
如果只希望发布主臂关节命令，不发布从臂 qpos，可执行：

```bash
conda activate aloha
cd /home/agilex/rhos_cobot
python -m scripts.post_collect.replay_data --dataset_dir ~/data --task_name aloha_mobile_dummy --only_pub_master --episode_idx 0
```

该模式会对主臂 action 做插值后以更高频率发布，便于下游节点平滑跟随。

## 4. 参数说明

- `--dataset_dir`：数据集根目录，例如 `~/data`
- `--task_name`：任务名，对应数据目录名
- `--episode_idx`：动作分块索引号，对应 `episode_{idx}.hdf5`
- `--only_pub_master`：是否只发布主臂关节姿态消息

默认 topic 如下：

- 图像：`/camera_f/color/image_raw`、`/camera_l/color/image_raw`、`/camera_r/color/image_raw`
- 主臂：`/master/joint_left`、`/master/joint_right`
- 从臂：`/puppet/joint_left`、`/puppet/joint_right`
- 底盘：`/cmd_vel`

如果需要末端位姿 replay，请使用 `python -m scripts.post_collect.replay_data_eef ...`。

## 5. Piper hybrid replay mock（VLM + pi0）

`scripts/run_piper_replay_mock.sh` 用 HDF5 数据集做离线观测回放，hybrid 模式下由 VLM 做任务拆解/重规划，pi0 负责 manipulate 动作推理。默认不会控制真实机器人。

### 5.1 带 progress checkpoint

适用于带 `assets/progress_metadata.json` 且 `has_progress_head=true` 的 checkpoint。下面命令使用 subtask progress 作为 `action["progress"]`，hybrid manipulate 会优先用 progress 判断当前子任务是否完成，VLM 负责更精确的 prompt replan 和 fallback。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u socks_proxy -u SOCKS_PROXY \
    -u no_proxy -u NO_PROXY \
    REPLAY_MODE=hybrid \
    TASK_NAME=pick_bread_leaf_3 \
    REPLAY_TASK_SPEC=none \
    START_TARGET=all \
    MANIPULATE_MAX_STEPS=64 \
    MANIPULATE_REPLAN_INTERVAL_STEPS=16 \
    POLICY_CONFIG=pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3 \
    CHECKPOINT_DIR=/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/openpi/checkpoints/pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3/pi05_pick_bread_leaf_progress_dual_20260424_061145/99999 \
    PROGRESS_SOURCE=subtask \
    PROGRESS_HEAD_MODE=auto \
    PLANNER_BACKEND=qz \
    QZ_STATE_FILE=config/vllm_server_state.json \
    QZ_USE_PROXY=0 \
    CUDA_VISIBLE_DEVICES=0 \
    bash scripts/run_piper_replay_mock.sh
```

关键参数：

- `PROGRESS_SOURCE=subtask`：OpenPI server 将 subtask progress 暴露为 `action["progress"]`。
- `PROGRESS_HEAD_MODE=auto`：replay/hybrid 根据 pi0 server metadata 自动启用 progress-first 完成检测。
- `REPLAY_TASK_SPEC=none`：只跑 bread/lettuce 三阶段时避免自动加载 `episode_4.hdf5` 的长任务 spec。

### 5.2 不带 progress checkpoint

不带 progress head 的 checkpoint 应关闭 progress-first 逻辑，让 hybrid manipulate 按固定间隔调用 VLM replanner。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u socks_proxy -u SOCKS_PROXY \
    -u no_proxy -u NO_PROXY \
    REPLAY_MODE=hybrid \
    TASK_NAME=pick_bread_leaf_3 \
    REPLAY_TASK_SPEC=none \
    START_TARGET=all \
    MANIPULATE_MAX_STEPS=64 \
    MANIPULATE_REPLAN_INTERVAL_STEPS=16 \
    POLICY_CONFIG=pi05_pick_bread_leaf_1+pick_bread_leaf_2+pick_bread_leaf_3 \
    CHECKPOINT_DIR=/path/to/non_progress_checkpoint \
    PROGRESS_SOURCE=task \
    PROGRESS_HEAD_MODE=off \
    PLANNER_BACKEND=qz \
    QZ_STATE_FILE=config/vllm_server_state.json \
    QZ_USE_PROXY=0 \
    CUDA_VISIBLE_DEVICES=0 \
    bash scripts/run_piper_replay_mock.sh
```

其中 `/path/to/non_progress_checkpoint` 需要替换为不带 progress head 的真实 checkpoint 目录。
