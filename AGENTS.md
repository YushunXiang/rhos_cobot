# rhos_cobot Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-24

## Active Technologies

- Python 3.11 with ROS1 `rospy` + `tyro`, `rospy`, `openpi_client`, `opencv-python`, `openai`, ROS message packages (`nav_msgs`, `geometry_msgs`, `sensor_msgs`), `cv_bridge` (001-llm-navigation-stage)

## Project Structure

```text
src/
tests/
```

## Commands

```bash
python -m scripts.collect.collect_data_eef_qpos --dataset_dir=./data --task_name <name> --max_timesteps 2500 --episode_idx <idx>
python -m scripts.post_collect.check_joints --dataset_dir ./data/ --data_key qpos
python -m examples.piper_real.main
PYTHONPATH=../openpi/packages/openpi-client/src ../openpi/.venv/bin/python -m pytest tests/test_replay_env.py
ruff check .
```

## Code Style

Python 3.11 with ROS1 `rospy`: Follow standard conventions

## Recent Changes

- 001-llm-navigation-stage: Added Python 3.11 with ROS1 `rospy` + `tyro`, `rospy`, `openpi_client`, `opencv-python`, `openai`, ROS message packages (`nav_msgs`, `geometry_msgs`, `sensor_msgs`), `cv_bridge`

<!-- MANUAL ADDITIONS START -->
## Project Overview

松灵 (AgileX) Cobot data collection and processing system for teleoperated dual-arm robots using the ALOHA architecture. Three-phase workflow: **collect -> post_collect -> deploy**.

- Data collection via ROS with multi-sensor fusion (3 RGB cameras, depth cameras, dual 7-DOF arms, mobile base)
- Post-collection data validation, visualization, and repair
- AI policy deployment via OpenPI inference client

## Setup

```bash
pip install -e .
conda activate aloha
```

## Common Commands

```bash
# Data collection (record mode: roslaunch with mode:=0)
sh scripts/init.sh
python -m scripts.collect.collect_data_eef_qpos --dataset_dir=./data --task_name <name> --max_timesteps 2500 --episode_idx <idx>

# Post-collection validation
python -m scripts.post_collect.check_joints --dataset_dir ./data/ --data_key qpos
python -m scripts.post_collect.visualize_episodes_eef --dataset_dir ./data/ --task_name <name> --episode_idx <idx>
python -m scripts.post_collect.replay_data_eef --dataset_dir ./data/ --task_name <name> --episode_idx <idx>
python -m scripts.post_collect.cal_time --dataset_dir ./data/ --task_name <name>
python -m scripts.post_collect.data_summary_simple

# Deploy (deploy mode: roslaunch with mode:=1, auto_enable:=true)
source examples/piper_real/.venv/bin/activate
python -m examples.piper_real.main
```

## Testing / Python Interpreter

Do not run bare `pytest` in this repo. The system `pytest` may be missing or may
bind to the wrong Python version, and `uv run` without an explicit interpreter can
create a Python 3.12 environment and rewrite `uv.lock`. Use the local OpenPI /
Piper Python 3.11 interpreter instead:

```bash
PYTHONPATH=../openpi/packages/openpi-client/src ../openpi/.venv/bin/python -m pytest tests/test_replay_env.py
```

If temporary dependencies are needed for an isolated run, pin the interpreter:

```bash
PYTHONPATH=../openpi/packages/openpi-client/src uv run --python 3.11 --with pytest --with opencv-python --with h5py --with numpy --with einops --with tyro --with pillow --with dm-tree --with openai --with websockets --with msgpack --with socksio pytest tests/test_replay_env.py
```

In hybrid mode the VLM replanner revises each manipulate subtask's prompt every
`MANIPULATE_REPLAN_INTERVAL_STEPS` policy steps and advances subtasks on the
replanner's `complete` signal; `MANIPULATE_MAX_STEPS` bounds each subtask.

All scripts are invoked as Python modules (`python -m scripts.<subdir>.<name>`).

## Architecture

### Core Package (`rhos_cobot/`)

- **`data_collection.py`**: `RosOperator` class for ROS subscriber-based multi-sensor data collection with deque-based frame synchronization. Subscribes to camera image topics, joint states, and odometry.
- **`utils.py`**: HDF5 load/save utilities and JPEG/PNG image compression/decompression.
- **`post_process.py`**: Video saving, frame visualization with action overlay, joint/EEF naming constants.

### Scripts (`scripts/`)

- **`collect/`**: Data collection scripts producing HDF5 episodes. User marks episodes as success (`s`), failed (`f`), or uncompleted (timeout).
- **`post_collect/`**: Validation (`check_joints.py`), visualization, replay, data repair (`fix_joints.py`), and naming normalization.

### Examples (`examples/`)

- **`piper_real/`**: Main deployment example using OpenPI client for inference. Has its own `.venv` and `requirements.txt`.
- Other directories (`aloha_real/`, `ur5/`, etc.) are integration examples for different robot platforms.

## Data Format

- **HDF5** files: `episode_{idx}.hdf5`
- Key paths: `/observations/qpos`, `/observations/qvel`, `/observations/effort`, `/observations/images/{cam}`, `/action`, `/action_eef`, `/base_action`
- 14 DOF joint space (7 per arm), images are JPEG-compressed
- Naming convention: `task_{task_id}_user_{user_id}_scene_{scene_id}`
- Episodes sorted into `success/`, `failed/`, `uncompleted/` subdirectories

## Key Dependencies

ROS (`rospy`, `sensor_msgs`, `geometry_msgs`), `h5py`, `numpy`, `opencv-python`, `tyro` (CLI args). Python 3.10+.

## Notes

- Documentation is primarily in Chinese.
- Hardware/ROS workflows are validated on the real robot; local pytest checks
  exist for replay/planner utilities and must use the Python 3.11 command above.
- ROS must be running before collection/replay scripts.
- Two robot modes: record (`mode:=0`, manual teleoperation) and deploy (`mode:=1`, autonomous with `auto_enable:=true`).
<!-- MANUAL ADDITIONS END -->
