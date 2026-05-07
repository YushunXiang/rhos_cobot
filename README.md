# rhos_cobot

松灵 cobot 数采及处理脚本

## 安装
```bash
pip install -e .
```

## 使用流程

详细参考 `docs/` 目录下的文档，使用顺序：**collect → post_collect → deploy**

### 1. 数据采集 (collect)

```bash
# 初始化环境
sh scripts/init.sh

# 激活环境，进入 record 模式（zsh 可用 init_record 快捷命令）
conda activate aloha
roslaunch piper start_ms_piper.launch mode:=0 auto_enabshle:=false

# 采集数据
python -m scripts.collect.collect_data_eef_qpos --dataset_dir=./data --task_name pick_all_zy --max_timesteps 2500 --episode_idx 29
```

采集结果存入 `episode_{idx}.hdf5`，采集结束时：
- 输入 `s`：成功，存入 `data_dir/task_name/`
- 输入 `f`：失败，存入 `data_dir/task_name/failed/`
- 超时：存入 `data_dir/task_name/uncompleted/`

详见 [docs/collect.md](docs/collect.md)

### 2. 数据后处理 (post_collect)

```bash
# 检查关节数据
python -m scripts.post_collect.check_joints --dataset_dir ./data/ --data_key qpos

# 修复离群点/突变（check 发现问题后）
python -m scripts.post_collect.fix_joints --dataset_dir ./data/ --task_name <task_name> --data_key qpos
# 修复结果在 data/<task_name>/fixed/，再次 check 验证

# 可视化数据
python -m scripts.post_collect.visualize_episodes_eef --dataset_dir ./data/ --task_name <task_name> --episode_idx 5

# 重播数据（可选，需先进入 deploy 模式）
python -m scripts.post_collect.replay_data --dataset_dir ./data/ --task_name <task_name> --episode_idx 5

# 网页复核 stage 标注（默认保存到 <dataset_dir>/fixed_stage/）
python -m scripts.post_collect.review_stage_web --dataset_dir ./ocl_data/pick_bread_leaf

# 计算时长
python -m scripts.post_collect.cal_time --dataset_dir ./data/ --task_name <task_name>

# 规范化数据命名（task_{task_id}_user_{user_id}_scene_{scene_id}）
python -m scripts.post_collect.data_summary_simple
```

详见 [docs/post_collect.md](docs/post_collect.md) 和 [docs/replay.md](docs/replay.md)

### 3. 部署推理 (deploy)

```bash
cp config/servers.example.toml config/servers.toml
# 按实际机器、模型路径和端口修改 config/servers.toml

# 初始化环境（拔下主臂航插线，重启插排）
sh scripts/init.sh

# 进入 deploy 模式（zsh 可用 init_deploy 快捷命令）
conda activate aloha
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# 运行推理
source examples/piper_real/.venv/bin/activate
python -m examples.piper_real.main
```

详见：

- 纯手臂部署（最小运行）：[docs/deploy.md](docs/deploy.md)
- VLM + VLA 双系统推理（含任务拆解、底盘导航、离线 replay）：[docs/dual_system_deploy.md](docs/dual_system_deploy.md)

## TODO
- [x] 数据检查代码 merge 进来
- [x] 离群点处理脚本 merge 进来 (待验证)
- [ ] openpi deploy merge 进来
- [ ] deploy readme 详细补充
