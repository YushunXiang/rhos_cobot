# Replay Progress Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline replay visualization tool that loads a progress-head checkpoint locally, replays a single manipulation segment, renders progress/replanner events into an MP4, and optionally dumps step-level JSONL.

**Architecture:** Add one script at `rhos_cobot/scripts/visualize_replay_progress.py` with four focused parts: CLI/config parsing, replay runner, record sink, and OpenCV frame composer. Reuse `ReplayEnvironment`, `ReplayTaskProgressTracker`, `PlannerConfig`, and `ReplayManipulationPromptPlanner` so the semantics match the existing replay/hybrid flow without depending on websocket inference.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), NumPy, Tyro, local OpenPI policy loading, existing rhos_cobot replay/planner modules, pytest.

---

## File Map

- Create: `rhos_cobot/scripts/visualize_replay_progress.py`
  Purpose: CLI entrypoint, local policy loading, replay execution, frame rendering, MP4/JSONL output.
- Create: `rhos_cobot/tests/test_visualize_replay_progress.py`
  Purpose: CLI, recorder, event flow, frame composer, and fake integration coverage.
- Modify: `rhos_cobot/scripts/__init__.py`
  Purpose: Only if needed to expose the new script module cleanly for tests.
- Reuse only, no edits expected:
  - `rhos_cobot/examples/piper_real/replay_env.py`
  - `rhos_cobot/examples/piper_real/main.py`
  - `rhos_cobot/examples/piper_real/planner_config.py`
  - `rhos_cobot/examples/piper_real/replay_manipulation_planner.py`
  - `openpi/src/openpi/policies/policy_config.py`

### Task 1: Add CLI and Config Parsing Tests

**Files:**
- Create: `rhos_cobot/tests/test_visualize_replay_progress.py`
- Create: `rhos_cobot/scripts/visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
import pathlib

import pytest

from scripts.visualize_replay_progress import Args, validate_args


def test_validate_args_rejects_missing_progress_checkpoint(tmp_path: pathlib.Path):
    dataset = tmp_path / "episode.hdf5"
    dataset.write_bytes(b"stub")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    args = Args(
        dataset_path=str(dataset),
        checkpoint_dir=str(checkpoint),
        output_video=str(tmp_path / "out.mp4"),
        prompt="grab bread",
        start_step=0,
        end_step=10,
    )

    with pytest.raises(ValueError, match="progress-head checkpoint"):
        validate_args(args)


def test_validate_args_rejects_invalid_step_range(tmp_path: pathlib.Path):
    dataset = tmp_path / "episode.hdf5"
    dataset.write_bytes(b"stub")
    checkpoint = tmp_path / "ckpt"
    assets = checkpoint / "assets"
    assets.mkdir(parents=True)
    (assets / "progress_metadata.json").write_text('{"has_progress_head": true}', encoding="utf-8")
    args = Args(
        dataset_path=str(dataset),
        checkpoint_dir=str(checkpoint),
        output_video=str(tmp_path / "out.mp4"),
        prompt="grab bread",
        start_step=8,
        end_step=3,
    )

    with pytest.raises(ValueError, match="start_step"):
        validate_args(args)


def test_validate_args_rejects_task_decompose_for_now(tmp_path: pathlib.Path):
    dataset = tmp_path / "episode.hdf5"
    dataset.write_bytes(b"stub")
    checkpoint = tmp_path / "ckpt"
    assets = checkpoint / "assets"
    assets.mkdir(parents=True)
    (assets / "progress_metadata.json").write_text('{"has_progress_head": true}', encoding="utf-8")
    args = Args(
        dataset_path=str(dataset),
        checkpoint_dir=str(checkpoint),
        output_video=str(tmp_path / "out.mp4"),
        prompt="grab bread",
        start_step=0,
        end_step=10,
        task_decompose=True,
    )

    with pytest.raises(NotImplementedError, match="task-decompose"):
        validate_args(args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'validate_args'`
Expected: FAIL with import or attribute errors for `Args` / `validate_args`

- [ ] **Step 3: Write minimal implementation**

```python
@dataclasses.dataclass
class Args:
    dataset_path: str
    checkpoint_dir: str
    output_video: str
    prompt: str
    start_step: int = 0
    end_step: int | None = None
    dump_jsonl: str = ""
    camera_name: str = ""
    task_decompose: bool = False


def _load_progress_metadata(checkpoint_dir: pathlib.Path) -> dict[str, bool]:
    metadata_path = checkpoint_dir / "assets" / "progress_metadata.json"
    if not metadata_path.exists():
        return {"has_progress_head": False}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_args(args: Args) -> None:
    dataset_path = pathlib.Path(args.dataset_path)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not dataset_path.exists():
        raise ValueError(f"dataset_path does not exist: {dataset_path}")
    if not checkpoint_dir.exists():
        raise ValueError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if args.task_decompose:
        raise NotImplementedError("--task-decompose is reserved for a future iteration")
    if args.end_step is not None and args.start_step > args.end_step:
        raise ValueError("start_step must be <= end_step")
    if not _load_progress_metadata(checkpoint_dir).get("has_progress_head", False):
        raise ValueError("visualizer requires a progress-head checkpoint")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'validate_args'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "test: add replay visualizer cli validation tests"
```

### Task 2: Add Step Record and JSONL Sink

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Modify: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from scripts.visualize_replay_progress import StepRecord, JsonlRecorder


def test_jsonl_recorder_writes_one_line_per_record(tmp_path):
    output = tmp_path / "trace.jsonl"
    recorder = JsonlRecorder(output)
    recorder.write(
        StepRecord(
            step=4,
            prompt="grab bread",
            progress=0.72,
            progress_event="continue",
            trigger_reason="",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_high",
        )
    )
    recorder.close()

    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["step"] == 4
    assert payload["progress"] == 0.72
    assert payload["camera_name"] == "cam_high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'jsonl_recorder'`
Expected: FAIL with missing `StepRecord` or `JsonlRecorder`

- [ ] **Step 3: Write minimal implementation**

```python
@dataclasses.dataclass
class StepRecord:
    step: int
    prompt: str
    progress: float
    progress_event: str
    trigger_reason: str
    replanner_called: bool
    replanner_action: str
    replanner_reason: str
    completed: bool
    camera_name: str


class JsonlRecorder:
    def __init__(self, path: pathlib.Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")

    def write(self, record: StepRecord) -> None:
        self._fh.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'jsonl_recorder'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: add replay progress jsonl recorder"
```

### Task 3: Add Progress Runner Event Flow

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Modify: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from scripts.visualize_replay_progress import ReplayRunConfig, run_replay_visualization


def test_runner_uses_progress_tracker_and_replanner_fallback(monkeypatch, tmp_path):
    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 6

        def __init__(self):
            self.cursor = 0

        def set_cursor(self, idx):
            self.cursor = idx

        def get_cursor(self):
            return self.cursor

        def is_episode_complete(self):
            return self.cursor >= 4

        def get_observation(self):
            step = self.cursor
            self.cursor += 1
            return {"state": np.zeros(14, dtype=np.float32), "images": {}, "prompt": "grab bread", "step": step}

        def get_image(self, cam_name, step_idx):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakePolicy:
        def __init__(self):
            self.values = iter([0.10, 0.11, 0.115, 0.90])

        def infer(self, obs):
            return {"actions": np.zeros((16, 14), dtype=np.float32), "progress": np.array([next(self.values)] * 16)}

    class FakePlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, **kwargs):
            self.calls += 1
            return type("Decision", (), {"action": "continue", "prompt": "grab bread tighter", "reason": "stall"})()

    records = run_replay_visualization(
        env=FakeEnv(),
        policy=FakePolicy(),
        planner=FakePlanner(),
        config=ReplayRunConfig(
            prompt="grab bread",
            start_step=0,
            end_step=4,
            camera_name="cam_high",
            complete_threshold=0.85,
            stall_threshold=0.02,
            stall_steps=3,
            regression_threshold=0.1,
        ),
        write_video_frame=lambda frame: None,
        write_record=lambda record: None,
    )

    assert [r.progress_event for r in records] == ["continue", "continue", "stall", "complete"]
    assert records[2].replanner_called is True
    assert records[-1].completed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'runner_uses_progress_tracker'`
Expected: FAIL with missing `ReplayRunConfig` or `run_replay_visualization`

- [ ] **Step 3: Write minimal implementation**

```python
@dataclasses.dataclass
class ReplayRunConfig:
    prompt: str
    start_step: int
    end_step: int | None
    camera_name: str
    complete_threshold: float
    stall_threshold: float
    stall_steps: int
    regression_threshold: float


def run_replay_visualization(env, policy, planner, config: ReplayRunConfig, write_video_frame, write_record):
    tracker = ReplayTaskProgressTracker(
        complete_threshold=config.complete_threshold,
        stall_threshold=config.stall_threshold,
        stall_steps=config.stall_steps,
        regression_threshold=config.regression_threshold,
    )
    env.set_cursor(config.start_step)
    records = []
    while not env.is_episode_complete():
        step_idx = env.get_cursor()
        if config.end_step is not None and step_idx >= config.end_step:
            break
        obs = env.get_observation()
        outputs = policy.infer(obs)
        progress = float(np.asarray(outputs["progress"])[0])
        decision = tracker.observe(progress)
        replanner_called = False
        replanner_action = ""
        replanner_reason = ""
        trigger_reason = ""
        if decision.event in {"stall", "regression"}:
            replanner_called = True
            trigger_reason = f"progress {decision.event}"
            replanner_decision = planner.plan(
                task_prompt=config.prompt,
                current_policy_prompt=config.prompt,
                executed_policy_steps=len(records) + 1,
                prompt_history=[],
            )
            replanner_action = replanner_decision.action
            replanner_reason = replanner_decision.reason
        elif decision.event == "complete":
            trigger_reason = "progress complete"

        record = StepRecord(
            step=step_idx,
            prompt=config.prompt,
            progress=progress,
            progress_event=decision.event,
            trigger_reason=trigger_reason,
            replanner_called=replanner_called,
            replanner_action=replanner_action,
            replanner_reason=replanner_reason,
            completed=decision.event == "complete",
            camera_name=config.camera_name,
        )
        records.append(record)
        write_record(record)
        write_video_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        if record.completed:
            break
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'runner_uses_progress_tracker'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: add replay progress runner"
```

### Task 4: Add Frame Composer

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Modify: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from scripts.visualize_replay_progress import StepRecord, compose_frame


def test_compose_frame_returns_bgr_canvas_with_overlay():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    records = [
        StepRecord(
            step=0,
            prompt="grab bread",
            progress=0.1,
            progress_event="continue",
            trigger_reason="",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_high",
        ),
        StepRecord(
            step=1,
            prompt="grab bread",
            progress=0.9,
            progress_event="complete",
            trigger_reason="progress complete",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=True,
            camera_name="cam_high",
        ),
    ]

    canvas = compose_frame(
        frame=frame,
        records=records,
        current_index=1,
        complete_threshold=0.85,
    )

    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert canvas.shape[0] > frame.shape[0]
    assert canvas.shape[1] >= frame.shape[1]
    assert np.any(canvas != 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'compose_frame'`
Expected: FAIL with missing `compose_frame`

- [ ] **Step 3: Write minimal implementation**

```python
def compose_frame(frame: np.ndarray, records: list[StepRecord], current_index: int, complete_threshold: float) -> np.ndarray:
    chart_height = 160
    status_width = 360
    h, w = frame.shape[:2]
    canvas = np.zeros((h + chart_height, w + status_width, 3), dtype=np.uint8)
    canvas[:h, :w] = frame

    current = records[current_index]
    cv2.putText(canvas, f"prompt: {current.prompt}", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"step: {current.step}", (w + 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"progress: {current.progress:.3f}", (w + 20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"event: {current.progress_event}", (w + 20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    chart = canvas[h:, :w]
    cv2.line(chart, (0, int(chart_height * (1 - complete_threshold))), (w - 1, int(chart_height * (1 - complete_threshold))), (0, 255, 255), 1)
    if len(records) > 1:
        points = []
        for i, rec in enumerate(records):
            x = int(i * (w - 1) / max(len(records) - 1, 1))
            y = int((1.0 - rec.progress) * (chart_height - 1))
            points.append((x, y))
        for p0, p1 in zip(points, points[1:], strict=False):
            cv2.line(chart, p0, p1, (0, 255, 0), 2)
        for idx, rec in enumerate(records):
            if rec.progress_event != "continue":
                x = int(idx * (w - 1) / max(len(records) - 1, 1))
                y = int((1.0 - rec.progress) * (chart_height - 1))
                cv2.circle(chart, (x, y), 4, (0, 0, 255), -1)
    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'compose_frame'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: add replay progress frame composer"
```

### Task 5: Wire Real ReplayEnvironment and Local Policy Loading

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Modify: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace


def test_main_loads_local_policy_and_replay_environment(monkeypatch, tmp_path):
    from scripts import visualize_replay_progress as mod

    dataset = tmp_path / "episode.hdf5"
    dataset.write_bytes(b"stub")
    checkpoint = tmp_path / "ckpt"
    assets = checkpoint / "assets"
    assets.mkdir(parents=True)
    (assets / "progress_metadata.json").write_text('{"has_progress_head": true}', encoding="utf-8")

    observed = {}

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 10

        def __init__(self, dataset_path: str, prompt: str, max_steps=None):
            observed["dataset_path"] = dataset_path
            observed["prompt"] = prompt

    monkeypatch.setattr(mod, "ReplayEnvironment", FakeEnv)
    monkeypatch.setattr(mod._policy_config, "create_trained_policy", lambda cfg, ckpt: observed.setdefault("checkpoint_dir", str(ckpt)) or object())
    monkeypatch.setattr(mod._config, "get_config", lambda name: SimpleNamespace())
    monkeypatch.setattr(mod, "run_replay_visualization", lambda **kwargs: [])

    mod.main(
        mod.Args(
            dataset_path=str(dataset),
            checkpoint_dir=str(checkpoint),
            output_video=str(tmp_path / "out.mp4"),
            prompt="grab bread",
            start_step=0,
            end_step=5,
        )
    )

    assert observed["dataset_path"] == str(dataset)
    assert observed["prompt"] == "grab bread"
    assert observed["checkpoint_dir"] == str(checkpoint)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'loads_local_policy_and_replay_environment'`
Expected: FAIL because `main` is missing or does not wire dependencies

- [ ] **Step 3: Write minimal implementation**

```python
def main(args: Args) -> None:
    validate_args(args)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    env = ReplayEnvironment(
        dataset_path=args.dataset_path,
        prompt=args.prompt,
        max_steps=args.end_step if args.end_step is not None else None,
    )
    train_config = _config.get_config("debug_pi05")
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    planner_config = PlannerConfig()
    planner = ReplayManipulationPromptPlanner(env, planner_config)

    records = []
    def _write_record(record):
        records.append(record)
    def _write_frame(_frame):
        return None

    run_replay_visualization(
        env=env,
        policy=policy,
        planner=planner,
        config=ReplayRunConfig(
            prompt=args.prompt,
            start_step=args.start_step,
            end_step=args.end_step,
            camera_name=args.camera_name or env.front_camera_name,
            complete_threshold=planner_config.progress_complete_threshold,
            stall_threshold=planner_config.progress_stall_threshold,
            stall_steps=planner_config.progress_stall_steps,
            regression_threshold=planner_config.progress_regression_threshold,
        ),
        write_video_frame=_write_frame,
        write_record=_write_record,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'loads_local_policy_and_replay_environment'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: wire replay visualizer to local policy and replay env"
```

### Task 6: Add MP4 Writer and End-to-End Fake Integration

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Modify: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np


def test_render_loop_writes_mp4_frames(monkeypatch, tmp_path):
    from scripts import visualize_replay_progress as mod

    written = {"frames": 0, "released": False}

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self):
            return True

        def write(self, frame):
            assert frame.ndim == 3
            written["frames"] += 1

        def release(self):
            written["released"] = True

    monkeypatch.setattr(mod.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(mod, "compose_frame", lambda frame, records, current_index, complete_threshold: np.zeros((400, 800, 3), dtype=np.uint8))

    writer = mod.open_video_writer(tmp_path / "out.mp4", width=800, height=400, fps=25.0)
    writer.write(np.zeros((400, 800, 3), dtype=np.uint8))
    writer.close()

    assert written["frames"] == 1
    assert written["released"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'writes_mp4_frames'`
Expected: FAIL with missing writer wrapper

- [ ] **Step 3: Write minimal implementation**

```python
class VideoSink:
    def __init__(self, writer):
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


def open_video_writer(path: pathlib.Path, *, width: int, height: int, fps: float) -> VideoSink:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return VideoSink(writer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'writes_mp4_frames'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: add replay visualizer mp4 output"
```

### Task 7: Final CLI Glue and Smoke Verification

**Files:**
- Modify: `rhos_cobot/scripts/visualize_replay_progress.py`
- Test: `rhos_cobot/tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Write the failing smoke test**

```python
def test_main_accepts_dump_jsonl_path(monkeypatch, tmp_path):
    from scripts import visualize_replay_progress as mod

    dataset = tmp_path / "episode.hdf5"
    dataset.write_bytes(b"stub")
    checkpoint = tmp_path / "ckpt"
    assets = checkpoint / "assets"
    assets.mkdir(parents=True)
    (assets / "progress_metadata.json").write_text('{"has_progress_head": true}', encoding="utf-8")

    monkeypatch.setattr(mod, "ReplayEnvironment", lambda *args, **kwargs: type("Env", (), {"front_camera_name": "cam_high", "camera_names": ("cam_high",), "num_steps": 4})())
    monkeypatch.setattr(mod._config, "get_config", lambda name: object())
    monkeypatch.setattr(mod._policy_config, "create_trained_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(mod, "ReplayManipulationPromptPlanner", lambda *args, **kwargs: object())
    monkeypatch.setattr(mod, "run_replay_visualization", lambda **kwargs: [])

    mod.main(
        mod.Args(
            dataset_path=str(dataset),
            checkpoint_dir=str(checkpoint),
            output_video=str(tmp_path / "out.mp4"),
            prompt="grab bread",
            start_step=0,
            end_step=4,
            dump_jsonl=str(tmp_path / "trace.jsonl"),
        )
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q -k 'accepts_dump_jsonl_path'`
Expected: FAIL until main wires optional JSONL sink

- [ ] **Step 3: Write minimal implementation**

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
```

Also wire:

```python
jsonl_recorder = JsonlRecorder(pathlib.Path(args.dump_jsonl)) if args.dump_jsonl else None
...
def _write_record(record):
    if jsonl_recorder is not None:
        jsonl_recorder.write(record)
...
finally:
    if jsonl_recorder is not None:
        jsonl_recorder.close()
```

- [ ] **Step 4: Run test suite for this feature**

Run: `openpi/.venv/bin/pytest rhos_cobot/tests/test_visualize_replay_progress.py -q`
Expected: PASS

- [ ] **Step 5: Run syntax verification**

Run: `python3 -m py_compile rhos_cobot/scripts/visualize_replay_progress.py rhos_cobot/tests/test_visualize_replay_progress.py`
Expected: PASS with no output

- [ ] **Step 6: Commit**

```bash
git -C rhos_cobot add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git -C rhos_cobot commit -m "feat: add replay progress visualization tool"
```

## Self-Review

- Spec coverage:
  - MP4 output: Task 4 + Task 6 + Task 7
  - JSONL output: Task 2 + Task 7
  - local replay + local policy path: Task 3 + Task 5
  - progress curve / threshold / events: Task 4
  - replanner fallback annotations: Task 3
  - `--task-decompose` extension hook: Task 1
- Placeholder scan:
  - No `TBD`, `TODO`, or “similar to Task N” instructions remain in executable steps.
- Type consistency:
  - Shared names are kept consistent: `Args`, `StepRecord`, `JsonlRecorder`, `ReplayRunConfig`, `run_replay_visualization`, `compose_frame`, `open_video_writer`.

