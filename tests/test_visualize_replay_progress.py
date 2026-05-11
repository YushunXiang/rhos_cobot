from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_checkpoint(tmp_path: Path, *, has_progress_head: bool | None) -> Path:
    checkpoint_dir = tmp_path / "checkpoint"
    assets_dir = checkpoint_dir / "assets"
    assets_dir.mkdir(parents=True)
    if has_progress_head is not None:
        (assets_dir / "progress_metadata.json").write_text(
            json.dumps({"has_progress_head": has_progress_head}),
            encoding="utf-8",
        )
    return checkpoint_dir


def _make_args(tmp_path: Path, *, checkpoint_dir: Path, **overrides):
    from scripts.visualize_replay_progress import Args

    values = {
        "dataset_path": str(tmp_path / "episode.hdf5"),
        "checkpoint_dir": str(checkpoint_dir),
        "output_video": str(tmp_path / "output.mp4"),
        "prompt": "pick up the red cup",
        "start_step": None,
        "end_step": None,
        "dump_jsonl": None,
        "camera_name": "cam_high",
        "task_decompose": False,
    }
    values.update(overrides)
    return Args(**values)


def test_parse_args_accepts_required_and_optional_flags():
    from scripts.visualize_replay_progress import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "--dataset-path",
            "/tmp/episode.hdf5",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--output-video",
            "/tmp/output.mp4",
            "--prompt",
            "pick up the red cup",
            "--start-step",
            "5",
            "--end-step",
            "18",
            "--dump-jsonl",
            "/tmp/trace.jsonl",
            "--camera-name",
            "cam_high",
            "--task-decompose",
        ]
    )

    assert ns.dataset_path == "/tmp/episode.hdf5"
    assert ns.checkpoint_dir == "/tmp/checkpoint"
    assert ns.output_video == "/tmp/output.mp4"
    assert ns.prompt == "pick up the red cup"
    assert ns.start_step == 5
    assert ns.end_step == 18
    assert ns.dump_jsonl == "/tmp/trace.jsonl"
    assert ns.camera_name == "cam_high"
    assert ns.task_decompose is True


def test_parse_args_accepts_video_font_path_flag():
    from scripts.visualize_replay_progress import parse_args

    args = parse_args(
        [
            "--dataset-path",
            "/tmp/episode.hdf5",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--output-video",
            "/tmp/output.mp4",
            "--prompt",
            "pick up the red cup",
            "--video-font-path",
            "/tmp/font.ttf",
        ]
    )

    assert args.video_font_path == Path("/tmp/font.ttf")


def test_validate_args_rejects_missing_dataset_path(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir)

    with pytest.raises(ValueError, match="dataset_path does not exist"):
        validate_args(args)


def test_validate_args_rejects_missing_checkpoint_dir(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = tmp_path / "missing-checkpoint"
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir)

    with pytest.raises(ValueError, match="checkpoint_dir does not exist"):
        validate_args(args)


def test_validate_args_rejects_missing_progress_metadata(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=None)
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir)

    with pytest.raises(ValueError, match="progress-head checkpoint"):
        validate_args(args)


def test_validate_args_rejects_false_progress_metadata(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=False)
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir)

    with pytest.raises(ValueError, match="progress-head checkpoint"):
        validate_args(args)


def test_validate_args_rejects_invalid_step_range(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir, start_step=8, end_step=3)

    with pytest.raises(ValueError, match="start_step must be <= end_step"):
        validate_args(args)


def test_validate_args_rejects_task_decompose(tmp_path: Path):
    from scripts.visualize_replay_progress import validate_args

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)
    args = _make_args(tmp_path, checkpoint_dir=checkpoint_dir, task_decompose=True)

    with pytest.raises(NotImplementedError, match="task-decompose"):
        validate_args(args)


def test_build_config_normalizes_paths_after_validation(tmp_path: Path):
    from scripts.visualize_replay_progress import build_config

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)
    args = _make_args(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        start_step=5,
        end_step=18,
        dump_jsonl=str(tmp_path / "trace.jsonl"),
        camera_name="cam_wrist",
    )

    config = build_config(args)

    assert config.dataset_path == dataset_path
    assert config.checkpoint_dir == checkpoint_dir
    assert config.output_video == tmp_path / "output.mp4"
    assert config.prompt == "pick up the red cup"
    assert config.start_step == 5
    assert config.end_step == 18
    assert config.dump_jsonl == tmp_path / "trace.jsonl"
    assert config.camera_name == "cam_wrist"
    assert config.task_decompose is False


def test_build_config_preserves_video_font_path(tmp_path: Path):
    from scripts.visualize_replay_progress import build_config

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)
    font_path = tmp_path / "font.ttf"
    font_path.write_bytes(b"stub")
    args = _make_args(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        video_font_path=font_path,
    )

    config = build_config(args)

    assert config.video_font_path == font_path


def test_step_record_exposes_expected_jsonl_fields():
    from scripts.visualize_replay_progress import StepRecord

    record = StepRecord(
        step=4,
        prompt="grab bread",
        progress=0.72,
        progress_horizon=(0.72, 0.74, 0.76),
        progress_horizon_mean=0.74,
        progress_horizon_last=0.76,
        expected_progress=0.40,
        progress_event="continue",
        trigger_reason="progress is steady",
        replanner_called=False,
        replanner_action="",
        replanner_reason="",
        completed=False,
        camera_name="cam_high",
    )

    payload = record.to_json_dict()

    assert payload == {
        "step": 4,
        "prompt": "grab bread",
        "progress": 0.72,
        "progress_horizon": [0.72, 0.74, 0.76],
        "progress_horizon_mean": 0.74,
        "progress_horizon_last": 0.76,
        "expected_progress": 0.40,
        "progress_event": "continue",
        "trigger_reason": "progress is steady",
        "replanner_called": False,
        "replanner_action": "",
        "replanner_reason": "",
        "completed": False,
        "camera_name": "cam_high",
    }


def test_jsonl_recorder_writes_one_json_object_per_line(tmp_path: Path):
    from scripts.visualize_replay_progress import JsonlRecorder, StepRecord

    output_path = tmp_path / "trace.jsonl"
    recorder = JsonlRecorder(output_path)
    recorder.write(
        StepRecord(
            step=4,
            prompt="grab bread",
            progress=0.72,
            progress_horizon=(0.72, 0.74, 0.76),
            progress_horizon_mean=0.74,
            progress_horizon_last=0.76,
            expected_progress=0.40,
            progress_event="continue",
            trigger_reason="progress is steady",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_high",
        )
    )
    recorder.close()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "step": 4,
        "prompt": "grab bread",
        "progress": 0.72,
        "progress_horizon": [0.72, 0.74, 0.76],
        "progress_horizon_mean": 0.74,
        "progress_horizon_last": 0.76,
        "expected_progress": 0.40,
        "progress_event": "continue",
        "trigger_reason": "progress is steady",
        "replanner_called": False,
        "replanner_action": "",
        "replanner_reason": "",
        "completed": False,
        "camera_name": "cam_high",
    }


def test_run_replay_visualization_uses_progress_tracker_and_falls_back_on_stall():
    from scripts.visualize_replay_progress import ReplayRunConfig, run_replay_visualization

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 6

        def __init__(self) -> None:
            self.cursor = 0
            self.cursor_history: list[int] = []

        def set_cursor(self, idx: int) -> None:
            self.cursor = idx
            self.cursor_history.append(idx)

        def get_cursor(self) -> int:
            return self.cursor

        def is_episode_complete(self) -> bool:
            return self.cursor >= 4

        def get_observation(self) -> dict[str, object]:
            step = self.cursor
            self.cursor += 1
            return {
                "state": step,
                "images": {},
                "prompt": "grab bread",
                "step": step,
            }

        def get_image(self, cam_name: str, step_idx: int):
            import numpy as np

            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakePolicy:
        def __init__(self) -> None:
            self.values = iter([0.10, 0.11, 0.115, 0.90])
            self.observations: list[dict[str, object]] = []

        def infer(self, obs: dict[str, object]) -> dict[str, object]:
            import numpy as np

            self.observations.append(obs)
            progress = next(self.values)
            return {
                "actions": np.zeros((16, 14), dtype=np.float32),
                "progress": np.array([progress] * 16, dtype=np.float32),
            }

    class FakePlanner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            return type(
                "Decision",
                (),
                {"action": "continue", "prompt": "grab bread tighter", "reason": "stall"},
            )()

    env = FakeEnv()
    policy = FakePolicy()
    planner = FakePlanner()
    records: list[object] = []
    frames: list[object] = []

    config = ReplayRunConfig(
        prompt="grab bread",
        start_step=0,
        end_step=4,
        camera_name="cam_high",
        complete_threshold=0.85,
        stall_threshold=0.02,
        stall_steps=3,
        regression_threshold=0.1,
    )

    result = run_replay_visualization(
        env=env,
        policy=policy,
        planner=planner,
        config=config,
        write_video_frame=frames.append,
        write_record=records.append,
    )

    assert [record.progress_event for record in result] == ["continue", "continue", "stall", "complete"]
    assert [record.progress for record in result] == pytest.approx([0.10, 0.11, 0.115, 0.90])
    assert result[2].replanner_called is True
    assert result[2].replanner_action == "continue"
    assert result[-1].completed is True
    assert len(planner.calls) == 1
    assert len(records) == 4
    assert len(frames) == 4


def test_run_replay_visualization_resets_local_policy_history_before_infer():
    from scripts.visualize_replay_progress import ReplayRunConfig, run_replay_visualization

    events: list[str] = []

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 1

        def __init__(self) -> None:
            self.cursor = 0

        def set_cursor(self, idx: int) -> None:
            self.cursor = idx

        def get_cursor(self) -> int:
            return self.cursor

        def is_episode_complete(self) -> bool:
            return self.cursor >= 1

        def get_observation(self) -> dict[str, object]:
            self.cursor += 1
            return {"step": 0}

    class FakePolicy:
        def reset_history(self) -> None:
            events.append("reset_history")

        def infer(self, _obs: dict[str, object]) -> dict[str, object]:
            import numpy as np

            events.append("infer")
            return {"progress": np.array([0.9], dtype=np.float32)}

    config = ReplayRunConfig(
        prompt="grab bread",
        start_step=0,
        end_step=1,
        camera_name="cam_high",
        complete_threshold=0.85,
        stall_threshold=0.02,
        stall_steps=3,
        regression_threshold=0.1,
    )

    run_replay_visualization(
        env=FakeEnv(),
        policy=FakePolicy(),
        planner=None,
        config=config,
        write_video_frame=None,
        write_record=None,
    )

    assert events == ["reset_history", "infer"]


def test_run_replay_visualization_can_confirm_completion_with_replanner():
    from scripts.visualize_replay_progress import ReplayRunConfig, run_replay_visualization

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 2

        def __init__(self) -> None:
            self.cursor = 0

        def set_cursor(self, idx: int) -> None:
            self.cursor = idx

        def get_cursor(self) -> int:
            return self.cursor

        def is_episode_complete(self) -> bool:
            return self.cursor >= 1

        def get_observation(self) -> dict[str, object]:
            self.cursor += 1
            return {"step": 0}

        def get_image(self, cam_name: str, step_idx: int):
            import numpy as np

            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakePolicy:
        def infer(self, obs: dict[str, object]) -> dict[str, object]:
            import numpy as np

            return {"progress": np.array([0.90], dtype=np.float32)}

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, **kwargs):
            self.calls += 1
            return type("Decision", (), {"action": "complete", "prompt": "", "reason": "done"})()

    result = run_replay_visualization(
        env=FakeEnv(),
        policy=FakePolicy(),
        planner=FakePlanner(),
        config=ReplayRunConfig(
            prompt="grab bread",
            start_step=0,
            end_step=1,
            camera_name="cam_high",
            complete_threshold=0.85,
            stall_threshold=0.02,
            stall_steps=3,
            regression_threshold=0.1,
            progress_confirm_with_replanner=True,
        ),
        write_video_frame=lambda frame: None,
        write_record=lambda record: None,
    )

    assert len(result) == 1
    assert result[0].progress_event == "complete"
    assert result[0].replanner_called is True
    assert result[0].replanner_action == "complete"
    assert result[0].completed is True


def test_compose_frame_returns_annotated_canvas_with_progress_pane():
    import numpy as np

    from scripts.visualize_replay_progress import StepRecord, compose_frame

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
            progress=0.42,
            progress_event="stall",
            trigger_reason="progress stalled",
            replanner_called=True,
            replanner_action="continue",
            replanner_reason="keep trying",
            completed=False,
            camera_name="cam_high",
        ),
        StepRecord(
            step=2,
            prompt="grab bread",
            progress=0.9,
            progress_event="complete",
            trigger_reason="progress threshold reached",
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
        current_index=2,
        complete_threshold=0.85,
    )

    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert canvas.shape[0] > frame.shape[0]
    assert canvas.shape[1] >= frame.shape[1]
    assert np.any(canvas != 0)


def test_compose_frame_preserves_camera_frame_in_top_left():
    import numpy as np

    from scripts.visualize_replay_progress import StepRecord, compose_frame

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = (13, 29, 47)
    records = [
        StepRecord(
            step=7,
            prompt="place the block",
            progress=0.66,
            progress_event="continue",
            trigger_reason="",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_wrist",
        )
    ]

    canvas = compose_frame(
        frame=frame,
        records=records,
        current_index=0,
        complete_threshold=0.75,
    )

    assert tuple(canvas[0, 0]) == (13, 29, 47)


def test_compose_frame_draws_complete_threshold_line():
    import numpy as np

    from scripts.visualize_replay_progress import StepRecord, compose_frame

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    records = [
        StepRecord(
            step=0,
            prompt="open drawer",
            progress=0.25,
            progress_event="continue",
            trigger_reason="",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_high",
        )
    ]

    canvas = compose_frame(
        frame=frame,
        records=records,
        current_index=0,
        complete_threshold=0.75,
    )

    progress_pane_y = frame.shape[0] + int(round((1.0 - 0.75) * 119))
    assert np.any(canvas[progress_pane_y, : frame.shape[1]] != 0)


def test_compose_frame_accepts_custom_font_path_via_env(tmp_path: Path, monkeypatch):
    """compose_frame must honor $RHOS_COBOT_VIDEO_FONT when no explicit font is passed."""
    import numpy as np

    from scripts.visualize_replay_progress import StepRecord, compose_frame

    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not dejavu.is_file():
        pytest.skip("DejaVuSans not available on this host")
    monkeypatch.setenv("RHOS_COBOT_VIDEO_FONT", str(dejavu))

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    records = [
        StepRecord(
            step=0,
            prompt="p",
            progress=0.4,
            progress_event="continue",
            trigger_reason="",
            replanner_called=False,
            replanner_action="",
            replanner_reason="",
            completed=False,
            camera_name="cam_high",
        )
    ]

    canvas = compose_frame(
        frame=frame,
        records=records,
        current_index=0,
        complete_threshold=0.85,
    )
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert np.any(canvas != 0)


def test_build_runtime_wires_replay_environment_policy_and_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from types import SimpleNamespace

    from scripts import visualize_replay_progress as mod

    dataset_path = tmp_path / "episode.hdf5"
    dataset_path.write_bytes(b"stub")
    checkpoint_dir = _make_checkpoint(tmp_path, has_progress_head=True)

    observed: dict[str, object] = {}

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 12

        def __init__(self, dataset_path: str, prompt: str, max_steps=None):
            observed["env_args"] = {
                "dataset_path": dataset_path,
                "prompt": prompt,
                "max_steps": max_steps,
            }

        def close(self) -> None:
            observed["env_closed"] = True

    class FakePlanner:
        def __init__(self, env, config):
            observed["planner_args"] = {"env": env, "config": config}

    train_config = SimpleNamespace(name="train-config")

    monkeypatch.setattr(mod, "ReplayEnvironment", FakeEnv)
    monkeypatch.setattr(mod, "ReplayManipulationPromptPlanner", FakePlanner)
    def _fake_get_config(name: str):
        observed["config_name"] = name
        return train_config

    monkeypatch.setattr(mod._config, "get_config", _fake_get_config)

    def _fake_create_trained_policy(cfg, ckpt, **kwargs):
        observed["policy_args"] = {"config": cfg, "checkpoint_dir": ckpt, "kwargs": kwargs}
        return object()

    monkeypatch.setattr(mod._policy_config, "create_trained_policy", _fake_create_trained_policy)

    runtime = mod.build_runtime(
        mod.Args(
            dataset_path=str(dataset_path),
            checkpoint_dir=str(checkpoint_dir),
            output_video=str(tmp_path / "output.mp4"),
            prompt="grab bread",
            start_step=3,
            end_step=9,
        )
    )

    assert observed["env_args"] == {
        "dataset_path": str(dataset_path),
        "prompt": "grab bread",
        "max_steps": 9,
    }
    assert observed["config_name"]
    assert observed["policy_args"]["config"] is train_config
    assert observed["policy_args"]["checkpoint_dir"] == checkpoint_dir
    assert observed["policy_args"]["kwargs"] == {"default_prompt": "grab bread"}
    assert observed["planner_args"]["env"] is runtime.env
    assert isinstance(runtime.planner_config, mod.PlannerConfig)
    assert runtime.run_config.start_step == 3
    assert runtime.run_config.end_step == 9
    assert runtime.run_config.camera_name == "cam_high"


def test_main_wires_cli_config_runtime_render_and_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import visualize_replay_progress as mod

    observed: dict[str, object] = {}

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True
            observed["env_closed"] = True

    runtime = mod.ReplayVisualizationRuntime(
        env=FakeEnv(),
        policy=object(),
        planner=object(),
        planner_config=mod.PlannerConfig(),
        run_config=mod.ReplayRunConfig(
            prompt="grab bread",
            start_step=0,
            end_step=4,
            camera_name="cam_high",
            complete_threshold=0.85,
            stall_threshold=0.02,
            stall_steps=3,
            regression_threshold=0.1,
        ),
    )
    config = mod.ReplayProgressVisualizationConfig(
        dataset_path=tmp_path / "episode.hdf5",
        checkpoint_dir=tmp_path / "checkpoint",
        output_video=tmp_path / "out.mp4",
        prompt="grab bread",
        start_step=0,
        end_step=4,
        dump_jsonl=tmp_path / "trace.jsonl",
        camera_name="cam_high",
        task_decompose=False,
    )

    def fake_build_config(args: mod.Args):
        observed["args"] = args
        return config

    def fake_build_runtime(config_arg):
        observed["config"] = config_arg
        return runtime

    def fake_render(runtime_arg, *, output_video, dump_jsonl, video_font_path=None):
        observed["render"] = {
            "runtime": runtime_arg,
            "output_video": output_video,
            "dump_jsonl": dump_jsonl,
            "video_font_path": video_font_path,
        }
        return []

    monkeypatch.setattr(mod, "build_config", fake_build_config)
    monkeypatch.setattr(mod, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(mod, "render_replay_visualization", fake_render)

    exit_code = mod.main(
        [
            "--dataset-path",
            str(config.dataset_path),
            "--checkpoint-dir",
            str(config.checkpoint_dir),
            "--output-video",
            str(config.output_video),
            "--prompt",
            "grab bread",
            "--start-step",
            "0",
            "--end-step",
            "4",
            "--dump-jsonl",
            str(config.dump_jsonl),
            "--camera-name",
            "cam_high",
        ]
    )

    assert exit_code == 0
    assert isinstance(observed["args"], mod.Args)
    assert observed["config"] is config
    assert observed["render"] == {
        "runtime": runtime,
        "output_video": config.output_video,
        "dump_jsonl": config.dump_jsonl,
        "video_font_path": config.video_font_path,
    }
    assert observed["env_closed"] is True


def test_open_video_writer_wraps_cv2_video_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import visualize_replay_progress as mod
    import numpy as np

    observed: dict[str, object] = {}

    class FakeWriter:
        def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):
            observed["args"] = {
                "path": path,
                "fourcc": fourcc,
                "fps": fps,
                "size": size,
            }
            self.released = False

        def isOpened(self) -> bool:
            return True

        def write(self, frame):
            observed["frame_shape"] = frame.shape

        def release(self) -> None:
            self.released = True
            observed["released"] = True

    monkeypatch.setattr(mod.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(mod.cv2, "VideoWriter_fourcc", lambda *codes: 1234)

    sink = mod.open_video_writer(tmp_path / "nested" / "out.mp4", width=800, height=400, fps=25.0)
    sink.write(np.zeros((400, 800, 3), dtype=np.uint8))
    sink.close()

    assert observed["args"] == {
        "path": str(tmp_path / "nested" / "out.mp4"),
        "fourcc": 1234,
        "fps": 25.0,
        "size": (800, 400),
    }
    assert observed["frame_shape"] == (400, 800, 3)
    assert observed["released"] is True


def test_render_replay_visualization_runs_runner_then_writes_video_and_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import visualize_replay_progress as mod
    import numpy as np

    class FakeEnv:
        front_camera_name = "cam_high"
        camera_names = ("cam_high",)
        num_steps = 4

        def __init__(self) -> None:
            self.cursor = 0
            self.image_requests: list[tuple[str, int]] = []

        def set_cursor(self, idx: int) -> None:
            self.cursor = idx

        def get_cursor(self) -> int:
            return self.cursor

        def is_episode_complete(self) -> bool:
            return self.cursor >= 4

        def get_observation(self) -> dict[str, object]:
            step = self.cursor
            self.cursor += 1
            return {"step": step}

        def get_image(self, cam_name: str, step_idx: int):
            self.image_requests.append((cam_name, step_idx))
            import numpy as np

            return np.full((120, 160, 3), step_idx * 40, dtype=np.uint8)

    class FakePolicy:
        def __init__(self) -> None:
            self.progresses = iter([0.10, 0.11, 0.115, 0.95])

        def infer(self, obs: dict[str, object]) -> dict[str, object]:
            import numpy as np

            progress = next(self.progresses)
            return {"progress": np.array([progress], dtype=np.float32)}

    class FakePlanner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            return type("Decision", (), {"action": "continue", "prompt": "refined", "reason": "stall"})()

    class FakeSink:
        def __init__(self) -> None:
            self.frames: list[tuple[int, int, int]] = []
            self.released = False

        def write(self, frame):
            self.frames.append(frame.shape)

        def close(self) -> None:
            self.released = True

    fake_sink = FakeSink()

    monkeypatch.setattr(mod, "open_video_writer", lambda path, *, width, height, fps: fake_sink)

    runtime = mod.ReplayVisualizationRuntime(
        env=FakeEnv(),
        policy=FakePolicy(),
        planner=FakePlanner(),
        planner_config=mod.PlannerConfig(),
        run_config=mod.ReplayRunConfig(
                prompt="pick up the red cup",
                start_step=0,
                end_step=4,
                camera_name="cam_high",
                complete_threshold=0.85,
                stall_threshold=0.02,
                stall_steps=3,
            regression_threshold=0.1,
            progress_confirm_with_replanner=False,
        ),
    )

    records = mod.render_replay_visualization(
        runtime,
        output_video=tmp_path / "out.mp4",
        dump_jsonl=tmp_path / "trace.jsonl",
        fps=12.5,
    )

    assert [record.step for record in records] == [0, 1, 2, 3]
    assert [record.progress_event for record in records] == ["continue", "continue", "stall", "complete"]
    assert fake_sink.frames
    assert fake_sink.released is True
    assert all(len(shape) == 3 for shape in fake_sink.frames)
    assert runtime.env.image_requests == [("cam_high", 0), ("cam_high", 1), ("cam_high", 2), ("cam_high", 3)]

    trace_lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 4
    assert json.loads(trace_lines[0])["step"] == 0
