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
