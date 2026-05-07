# Post Collection Scripts

采集完数据后，使用以下脚本进行验证以及数据处理。

## 0. 前后版本差异问题
由于采集脚本的版本变化问题，导致部分采集到的数据并没有正确标注 compress 标志位。


## 1. 检查关节数据
使用脚本检查关节数据是否存在数值溢出或者数据丢失。

```bash
python -m scripts.post_collect.check_joints --dataset_dir ./data/ --data_key qpos [--task_name task0063_user0012_scene0004_ep0]
```
task_name 可选，指定检查某个任务，否则检查所有任务。

此脚本一共检查四个方面：
1. 关节数据是否存在 全部为0 的情况
2. 关节数据是否存在 数值溢出导致的异常值（大于 π 或小于 -π）
3. 关节数据是否存在 突变（相邻两帧数据变化过大，默认阈值为 1 弧度）
4. 压缩标志位 compress 是否正确

检查日志将输出到控制台和指定 log 文件中，请仔细查看是否存在警告信息。

如果 check_joints 发现了离群点或突变，可以使用 fix_joints 脚本进行修复，完整流程如下：

**步骤一：确认问题（action 同理）**
```bash
python -m scripts.post_collect.check_joints --dataset_dir ./data/ --data_key qpos --task_name <task_name>
```
查看日志 `data/check_log/check_qpos.log`，确认存在 WARNING。

**步骤二：运行修复**
```bash
python -m scripts.post_collect.fix_joints --dataset_dir ./data/ --data_key qpos --task_name <task_name>
```
脚本对每个关节逐维度做线性插值（前后帧均值替换异常值），先处理超出 `[-π, π]` 的点，再处理突变点。修复后的数据**不覆盖原始文件**，存入 `data/<task_name>/fixed/` 子目录。修复日志写入 `data/fix_log/fix_qpos.log`。

**步骤三：验证修复结果**
```bash
python -m scripts.post_collect.check_joints --dataset_dir ./data/<task_name>/fixed/ --data_key qpos
```
若日志中不再有 WARNING，说明修复有效。

**步骤四（可选）：可视化对比**
```bash
python -m scripts.post_collect.visualize_episodes_eef --dataset_dir ./data/<task_name>/fixed/ --task_name fixed --episode_idx <idx>
```
对比修复前后的关节曲线，确认异常点已被平滑。

> 注意：fix_joints 只能通过插值修复离群点和突变点，无法解决数据全为零、视频损坏等其他问题。对于严重损坏的 episode，建议直接丢弃。

## 2. 可视化数据
可视化采集到的视频、关节角度、末端执行器等数据。  
For example:

```bash
python -m scripts.post_collect.visualize_episodes --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5
```
该脚本会在指定的目录下生成可视化结果，包括视频、关节角度图像等。请仔细观察是否存在视频数据损坏、关节角度记录异常（如左右臂数据明显颠倒）、末端执行器数据异常等问题。

## 2.1 Stage 网页复核
对于新增 `stage`、缺失 `stage` 或者怀疑录制时漏按 `space` 的数据，可以启动本地网页进行人工复核与修正。该工具会读取指定目录下的 `episode_*.hdf5`，显示相机画面、时间轴和阶段切换点；默认把修正结果写到 `fixed_stage/` 副本，也可以显式开启覆盖原文件。

```bash
# 默认：保存到 <dataset_dir>/fixed_stage/
python -m scripts.post_collect.review_stage_web --dataset_dir ./ocl_data/pick_bread_leaf

# 如需允许直接覆盖原始 HDF5，再显式加开关
python -m scripts.post_collect.review_stage_web --dataset_dir ./ocl_data/pick_bread_leaf --allow_overwrite
```

打开浏览器访问脚本启动时打印的地址，推荐先筛选异常 episode：
- `missing_stage`：源文件没有 `/stage` dataset
- `all_zero_stage`：整段都还是 stage 0，常见于漏按 `space`
- `missing_qpos` / `empty_images_group`：结构不完整，仅允许查看，不允许保存

网页内支持：
- 拖动时间轴边界调整阶段切换点
- 按区间批量重写 stage
- 修改当前 segment 的 stage 值
- 保存到 `fixed_stage/episode_xxx.hdf5`，并在下次打开时优先读取 fixed 副本继续编辑

## 3. 重播数据（可选）
在可视化检查通过后，可以使用以下脚本重播数据，验证数据的完整性和正确性。  
⚠️Warning⚠️  运行重播前请确保：  
重播数据需要确保采集时的环境和设备状态与实际使用时一致;
机器处于 deploy模式，即可以接收关节角度和末端执行器位置的指令，详见 [deploy模式](./deploy.md)。

```bash
# 推荐：按 cobot_magic 的 joint replay 习惯重播图像、主臂 action 和从臂 qpos
python -m scripts.post_collect.replay_data --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5

# 如需末端位姿指令重播，继续使用 eef 版本
python -m scripts.post_collect.replay_data_eef --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5

# 保留旧的 joint 入口
python -m scripts.post_collect.replay_data_joint --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5
```
推荐优先阅读 [replay.md](./replay.md)，其中包含断电重启、拔主臂航插头、启动顺序以及 `--only_pub_master` 的使用说明。
这些脚本会重播指定的 episode 数据，并在控制台输出关节角度、末端位姿或其他相关状态信息。

## 4. 数据处理
如果数据可视化和重播都通过了，可以使用以下脚本进行数据处理，对数据命名以及元数据进行格式统一。

### 4.1 计算时长
```bash
# 计算该任务总时长，填入飞书表格，相机频率默认为25hz，即25步为1s，修改相机频率后请更改 --camera_fps
python -m scripts.post_collect.cal_time --dataset_dir ./data/ --task_name task74_ep0003
```
### 4.2 规范化数据名
统一数据命名为**task_{task_id}_user_{user_id}_scene_{scene_id}**  
task_id：对应飞书表格前面序号  
user_id：对应飞书表格分配的序号  
scene_id：对应同一个任务不同场景的数据（一个任务可能在不同任务场景，或者改变一些变量多次采集）  

```bash
# 规范化数据名
python -m scripts.post_collect.data_summary_simple
```
要求输入
1. 处理的文件夹路径，即刚刚采集完的所有episode_{idx}.hdf5文件所在路径
2. task_id， user_id， scene_id

### 4.3 生成json存储元数据

## 5. 统一上传
将处理好的数据上传到服务器，上传路径详见飞书文档。
