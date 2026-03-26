# HDF5 Replay Debug Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable offline debugging of the inference pipeline by replaying HDF5 dataset observations instead of requiring live robot hardware and ROS.

**Architecture:** Create a `ReplayEnvironment` that implements the `openpi_client.runtime.environment.Environment` interface, reading observations from HDF5 episode files. The existing `main.py` gains a `--replay-dataset` flag; when set, it substitutes `ReplayEnvironment` for `PiperRealEnvironment`, skips ROS initialization, safety confirmation, and navigation, and logs predicted actions for comparison with recorded ground-truth actions. The `Runtime._step()` episode completion check (currently commented out) must be re-enabled for replay termination to work.

**Tech Stack:** h5py, numpy, cv2, einops, openpi_client, dm_env (existing deps only)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `examples/piper_real/replay_env.py` | **Create** | `ReplayEnvironment` class — reads HDF5, serves observations in openpi format |
| `examples/piper_real/main.py` | **Modify** | Add `--replay-dataset` CLI arg, branch to replay path, add `import numpy as np` |
| `tests/test_replay_env.py` | **Create** | Unit tests for `ReplayEnvironment` |

**External dependency (not in this repo):**
| File | Action | Responsibility |
|------|--------|----------------|
| `openpi-client: runtime/runtime.py:88-91` | **Modify** | Uncomment `is_episode_complete()` / `max_episode_steps` check in `_step()` |

---

### Task 0: Re-enable episode completion check in Runtime

**CRITICAL:** The `Runtime._step()` method at `/home/agilex/cobot_magic/openpi/packages/openpi-client/src/openpi_client/runtime/runtime.py` has the episode completion check commented out (lines 88-91). Without this, `is_episode_complete()` is never called and the replay loop will run forever, eventually crashing with `IndexError` when the dataset is exhausted.

**Files:**
- Modify: `/home/agilex/cobot_magic/openpi/packages/openpi-client/src/openpi_client/runtime/runtime.py:88-91`

- [ ] **Step 1: Uncomment the episode completion check**

Change lines 88-91 from:
```python
        # if self._environment.is_episode_complete() or (
        #     self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        # ):
        #     self.mark_episode_complete()
```

To:
```python
        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
```

- [ ] **Step 2: Verify existing live mode still works**

This change re-enables a code path that was intentionally part of the Runtime design. The existing `PiperRealEnvironment.is_episode_complete()` always returns `False`, so live mode behavior is unchanged. The `max_episode_steps` check now also takes effect — this is desirable as it provides an upper bound on episode length.

- [ ] **Step 3: Commit**

```bash
cd /home/agilex/cobot_magic/openpi
git add packages/openpi-client/src/openpi_client/runtime/runtime.py
git commit -m "fix(runtime): re-enable episode completion check in Runtime._step()"
```

---

### Task 1: Create ReplayEnvironment

**Files:**
- Create: `examples/piper_real/replay_env.py`
- Test: `tests/test_replay_env.py`

- [ ] **Step 1: Write the failing test for ReplayEnvironment initialization**

```python
# tests/test_replay_env.py
"""Tests for HDF5 replay environment."""

import cv2
import h5py
import numpy as np
import pytest


def _create_test_hdf5(path: str, num_steps: int = 10, *, compressed: bool = True) -> str:
    """Create a minimal HDF5 episode file for testing."""
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["compress"] = compressed
        f.attrs["fps"] = 25

        qpos = np.random.randn(num_steps, 14).astype(np.float64)
        f.create_dataset("/observations/qpos", data=qpos)
        f.create_dataset("/observations/qvel", data=np.zeros((num_steps, 14)))
        f.create_dataset("/observations/effort", data=np.zeros((num_steps, 14)))
        f.create_dataset("/action", data=np.random.randn(num_steps, 14).astype(np.float64))
        f.create_dataset("/base_action", data=np.zeros((num_steps, 2)))

        for cam_name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if compressed:
                dt = h5py.special_dtype(vlen=np.uint8)
                ds = f.create_dataset(
                    f"/observations/images/{cam_name}",
                    shape=(num_steps,),
                    dtype=dt,
                )
                for i in range(num_steps):
                    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    _, buf = cv2.imencode(".jpg", img)
                    ds[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
            else:
                imgs = np.random.randint(0, 255, (num_steps, 480, 640, 3), dtype=np.uint8)
                f.create_dataset(f"/observations/images/{cam_name}", data=imgs)
    return path


class TestReplayEnvironment:
    def test_init_loads_episode(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        assert env._num_steps == 5

    def test_reset_sets_cursor_to_zero(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        assert env._cursor == 0

    def test_get_observation_returns_correct_format(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        obs = env.get_observation()

        assert "state" in obs
        assert "images" in obs
        assert "prompt" in obs
        assert obs["state"].shape == (14,)
        assert obs["prompt"] == "test task"
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert cam in obs["images"]
            assert obs["images"][cam].shape == (3, 224, 224)  # CHW
            assert obs["images"][cam].dtype == np.uint8

    def test_get_observation_advances_cursor(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.get_observation()
        assert env._cursor == 1

    def test_is_episode_complete_at_end(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=2)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        assert not env.is_episode_complete()
        env.get_observation()
        assert not env.is_episode_complete()
        env.get_observation()
        assert env.is_episode_complete()

    def test_apply_action_logs_action(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=3)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.get_observation()

        fake_action = {"actions": np.zeros(14)}
        env.apply_action(fake_action)
        assert len(env.predicted_actions) == 1

    def test_ground_truth_actions_accessible(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        assert env.ground_truth_actions.shape == (5, 14)

    def test_uncompressed_images(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=3, compressed=False)

        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        obs = env.get_observation()

        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert obs["images"][cam].shape == (3, 224, 224)
            assert obs["images"][cam].dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/agilex/rhos_cobot-001-llm-navigation-stage && python -m pytest tests/test_replay_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.piper_real.replay_env'`

- [ ] **Step 3: Implement ReplayEnvironment**

```python
# examples/piper_real/replay_env.py
"""HDF5 replay environment for offline inference debugging.

Implements the openpi_client Environment interface by reading pre-recorded
observations from an HDF5 episode file instead of querying live hardware.
Actions predicted by the policy are collected for comparison with the
ground-truth actions stored in the dataset.
"""

import logging
from typing import Optional

import cv2
import einops
import h5py
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class ReplayEnvironment(_environment.Environment):
    """Replays observations from an HDF5 episode file.

    Parameters
    ----------
    dataset_path:
        Absolute path to a single ``episode_*.hdf5`` file.
    prompt:
        Task description forwarded to the policy.
    render_height, render_width:
        Target image resolution (must match policy training config).
    """

    def __init__(
        self,
        dataset_path: str,
        prompt: str = "",
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._cursor: int = 0

        # --- load dataset ---------------------------------------------------
        with h5py.File(dataset_path, "r") as f:
            compressed = f.attrs.get("compress", False)
            self._qpos: np.ndarray = f["/observations/qpos"][()]
            self._num_steps: int = self._qpos.shape[0]

            self.ground_truth_actions: np.ndarray = f["/action"][()]
            self.ground_truth_base_actions: Optional[np.ndarray] = (
                f["/base_action"][()] if "base_action" in f else None
            )

            # Decode images eagerly — dataset fits in memory for typical
            # episode lengths (< 2000 steps).
            self._images: dict[str, list[np.ndarray]] = {}
            for cam_name in f["/observations/images"].keys():
                if "_depth" in cam_name:
                    continue
                raw = f[f"/observations/images/{cam_name}"][()]
                frames: list[np.ndarray] = []
                for frame_data in raw:
                    if compressed:
                        img = cv2.imdecode(
                            np.frombuffer(frame_data, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                    else:
                        img = frame_data
                    frames.append(img)
                self._images[cam_name] = frames

        self.predicted_actions: list[np.ndarray] = []

        logging.info(
            "ReplayEnvironment loaded %s (%d steps, cameras: %s)",
            dataset_path,
            self._num_steps,
            list(self._images.keys()),
        )

    # -- Environment interface ------------------------------------------------

    @override
    def reset(self) -> None:
        self._cursor = 0
        self.predicted_actions = []

    @override
    def is_episode_complete(self) -> bool:
        return self._cursor >= self._num_steps

    @override
    def get_observation(self) -> dict:
        idx = self._cursor
        if idx >= self._num_steps:
            raise IndexError(f"Replay exhausted at step {idx}/{self._num_steps}")

        images: dict[str, np.ndarray] = {}
        for cam_name, frames in self._images.items():
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    frames[idx], self._render_height, self._render_width
                )
            )
            images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        self._cursor += 1

        return {
            "state": self._qpos[idx].copy(),
            "images": images,
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        if "actions" in action:
            self.predicted_actions.append(np.array(action["actions"]))
            step_idx = len(self.predicted_actions) - 1
            logging.info(
                "Replay step %d — predicted action[:4]: %s",
                step_idx,
                action["actions"][:4],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/agilex/rhos_cobot-001-llm-navigation-stage && python -m pytest tests/test_replay_env.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add examples/piper_real/replay_env.py tests/test_replay_env.py
git commit -m "feat(replay): add ReplayEnvironment for offline HDF5 debugging"
```

---

### Task 2: Integrate ReplayEnvironment into main.py

**Files:**
- Modify: `examples/piper_real/main.py`
- Test: `tests/test_replay_env.py` (append integration test)

- [ ] **Step 1: Write the failing test for CLI integration**

Append to `tests/test_replay_env.py`:

```python
class TestMainReplayIntegration:
    """Verify that main.py accepts --replay-dataset and constructs ReplayEnvironment."""

    def test_args_has_replay_dataset_field(self):
        import dataclasses
        from examples.piper_real.main import Args

        fields = {f.name for f in dataclasses.fields(Args)}
        assert "replay_dataset" in fields

    def test_args_replay_dataset_default_is_empty(self):
        from examples.piper_real.main import Args

        args = Args()
        assert args.replay_dataset == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/agilex/rhos_cobot-001-llm-navigation-stage && python -m pytest tests/test_replay_env.py::TestMainReplayIntegration -v`
Expected: FAIL — `AssertionError` (field does not exist yet)

- [ ] **Step 3: Add `--replay-dataset` arg and replay branch to main.py**

First, add `import numpy as np` to the imports at the top of `main.py`:

```python
import dataclasses
import logging

import numpy as np
import tyro
```

Then add the field to `Args`:

```python
@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"  # H100
    port: int = 9000
    action_horizon: int = 16
    num_episodes: int = 1
    max_episode_steps: int = 1000
    save_log: bool = False
    prompt: str = ""
    use_llm_planner: bool = False
    use_robot_base: bool = False
    replay_dataset: str = ""  # Path to HDF5 episode file for offline replay
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)
```

Then modify `main()` to branch on `replay_dataset`. The full updated `main()`:

```python
def main(args: Args) -> None:
    prompt = args.prompt.strip()

    # ── Replay mode: skip ROS, safety, navigation ──────────────────────
    if args.replay_dataset:
        from examples.piper_real import replay_env as _replay_env

        logging.info("Replay mode: loading %s", args.replay_dataset)
        environment = _replay_env.ReplayEnvironment(
            dataset_path=args.replay_dataset,
            prompt=prompt,
        )

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=50,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()

        # ── Post-replay summary ────────────────────────────────────────
        if environment.predicted_actions:
            predicted = np.stack(environment.predicted_actions)
            gt = environment.ground_truth_actions[: len(predicted)]
            mae = np.mean(np.abs(predicted[:, :14] - gt[:, :14]))
            logging.info(
                "Replay finished: %d steps, MAE vs ground-truth: %.6f",
                len(predicted),
                mae,
            )
        return

    # ── Live mode (original flow, unchanged) ───────────────────────────
    navigation_requested = args.use_llm_planner and bool(prompt)
    base_motion_requested = args.use_robot_base or navigation_requested

    if base_motion_requested:
        args.planner.validate_motion_limits()
    if navigation_requested:
        args.planner.validate_service_config()

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    if args.save_log:
        _logger.InputJointStateLogger()
        _logger.OutputJointStateLogger()

    environment = _env.PiperRealEnvironment(
        reset_position=metadata.get("reset_pose"),
        prompt=args.prompt,
        use_robot_base=args.use_robot_base,
        max_base_linear_vel=args.planner.max_linear_vel,
        max_base_angular_vel=args.planner.max_angular_vel,
    )

    if base_motion_requested and not _base_safety.confirm_base_motion_safety(
        prompt,
        use_llm_planner=navigation_requested,
        use_robot_base=args.use_robot_base,
    ):
        _base_safety.stop_base(environment.ros_operator)
        logging.error("Base motion aborted before execution; manipulation will not start.")
        return

    if args.use_llm_planner:
        if prompt:
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)
            if not planner.run(task_prompt=prompt):
                _base_safety.stop_base(environment.ros_operator)
                logging.error("Navigation failed; manipulation will not start.")
                return
            logging.info("Navigation succeeded; starting manipulation runtime.")
        else:
            logging.info("Navigation enabled but prompt is empty; skipping navigation stage.")
            logging.info('Navigation status: {"status": "navigation_skipped", "reason": "empty prompt"}')
            logging.info("Navigation skipped; starting manipulation runtime.")
    else:
        logging.info("Navigation skipped because use_llm_planner is false.")
        logging.info('Navigation status: {"status": "navigation_skipped", "reason": "use_llm_planner is false"}')
        logging.info("Navigation skipped; starting manipulation runtime.")

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        runtime.run()
    finally:
        if args.use_llm_planner or args.use_robot_base:
            _base_safety.stop_base(environment.ros_operator)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/agilex/rhos_cobot-001-llm-navigation-stage && python -m pytest tests/test_replay_env.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add examples/piper_real/main.py
git commit -m "feat(replay): integrate --replay-dataset flag into main.py"
```

---

### Task 3: Update deploy.md documentation

**Files:**
- Modify: `docs/deploy.md`

- [ ] **Step 1: Add replay/debug section to deploy.md**

Append after section 7 (验证), before the end of the file:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add docs/deploy.md
git commit -m "docs: add offline replay debug section to deploy guide"
```

---

### Task 4: End-to-end manual verification

- [ ] **Step 1: Verify replay environment loads the real dataset**

Run:
```bash
cd /home/agilex/rhos_cobot-001-llm-navigation-stage
source examples/piper_real/.venv/bin/activate
python -c "
from examples.piper_real.replay_env import ReplayEnvironment
env = ReplayEnvironment(
    dataset_path='/home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5',
    prompt='turn on the water tap.',
)
env.reset()
obs = env.get_observation()
print('state shape:', obs['state'].shape)
print('prompt:', obs['prompt'])
for k, v in obs['images'].items():
    print(f'{k}: shape={v.shape}, dtype={v.dtype}')
print('total steps:', env._num_steps)
print('episode complete:', env.is_episode_complete())
"
```

Expected output:
```
state shape: (14,)
prompt: turn on the water tap.
cam_high: shape=(3, 224, 224), dtype=uint8
cam_left_wrist: shape=(3, 224, 224), dtype=uint8
cam_right_wrist: shape=(3, 224, 224), dtype=uint8
total steps: 817
episode complete: False
```

- [ ] **Step 2: Run full replay with inference server (requires server running)**

Run:
```bash
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --replay-dataset /home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5 \
  --prompt "turn on the water tap."
```

Expected: Logs show each step's predicted action and a final MAE summary.

- [ ] **Step 3: Commit final state if any fixes were needed**
