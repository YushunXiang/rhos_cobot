#!/usr/bin/env python3
"""Offline replay progress visualizer CLI."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import cv2
import numpy as np


def _log(msg: str) -> None:
    print(f"[visualize_replay_progress] {time.strftime('%H:%M:%S')} {msg}", flush=True)
    sys.stdout.flush()


DEFAULT_OUTPUT_DIR = Path(
    "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/tmp"
)


def _default_output_video(dataset_path: str) -> str:
    return str(DEFAULT_OUTPUT_DIR / f"{Path(dataset_path).stem}_progress.mp4")


def _default_dump_jsonl(dataset_path: str) -> str:
    return str(DEFAULT_OUTPUT_DIR / f"{Path(dataset_path).stem}_progress.jsonl")

from examples.piper_real.planner_config import PlannerConfig
from examples.piper_real.replay_env import ReplayEnvironment
from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclass(frozen=True)
class ReplayProgressDecision:
    event: str
    progress: float
    detail: str = ""


class ReplayTaskProgressTracker:
    def __init__(
        self,
        *,
        complete_threshold: float,
        stall_threshold: float,
        stall_steps: int,
        regression_threshold: float,
    ) -> None:
        self._complete_threshold = complete_threshold
        self._stall_threshold = stall_threshold
        self._stall_steps = stall_steps
        self._regression_threshold = regression_threshold
        self._values: list[float] = []
        self._max_progress: float = 0.0

    def observe(self, progress: float) -> ReplayProgressDecision:
        value = float(progress)
        self._values.append(value)
        self._max_progress = max(self._max_progress, value)

        if value >= self._complete_threshold:
            return ReplayProgressDecision("complete", value, "progress threshold reached")

        if value < self._max_progress - self._regression_threshold:
            return ReplayProgressDecision("regression", value, "progress regressed")

        if len(self._values) >= self._stall_steps:
            window = self._values[-self._stall_steps :]
            if max(window) - min(window) < self._stall_threshold:
                return ReplayProgressDecision("stall", value, "progress stalled")

        return ReplayProgressDecision("continue", value)


@dataclass(frozen=True)
class Args:
    dataset_path: str
    checkpoint_dir: str
    output_video: str
    prompt: str
    start_step: int | None = None
    end_step: int | None = None
    dump_jsonl: str | None = None
    camera_name: str = "cam_high"
    task_decompose: bool = False
    no_replanner: bool = False


@dataclass(frozen=True)
class ReplayRunConfig:
    prompt: str
    start_step: int
    end_step: int | None
    camera_name: str
    complete_threshold: float
    stall_threshold: float
    stall_steps: int
    regression_threshold: float
    progress_confirm_with_replanner: bool = False
    enable_replanner: bool = True


@dataclass(frozen=True)
class ReplayProgressVisualizationConfig:
    dataset_path: Path
    checkpoint_dir: Path
    output_video: Path
    prompt: str
    start_step: int | None
    end_step: int | None
    dump_jsonl: Path | None
    camera_name: str
    task_decompose: bool
    no_replanner: bool = False


@dataclass(frozen=True)
class ReplayVisualizationRuntime:
    env: Any
    policy: _ReplayRunPolicyProtocol
    planner: Any
    planner_config: PlannerConfig
    run_config: ReplayRunConfig


class _ReplayRunPolicyProtocol:
    def infer(self, obs: Any) -> dict[str, Any]:  # pragma: no cover - protocol only
        raise NotImplementedError


@dataclass(frozen=True)
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
    progress_horizon: tuple[float, ...] = ()
    progress_horizon_mean: float = 0.0
    progress_horizon_last: float = 0.0
    expected_progress: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "prompt": self.prompt,
            "progress": self.progress,
            "progress_horizon": list(self.progress_horizon),
            "progress_horizon_mean": self.progress_horizon_mean,
            "progress_horizon_last": self.progress_horizon_last,
            "expected_progress": self.expected_progress,
            "progress_event": self.progress_event,
            "trigger_reason": self.trigger_reason,
            "replanner_called": self.replanner_called,
            "replanner_action": self.replanner_action,
            "replanner_reason": self.replanner_reason,
            "completed": self.completed,
            "camera_name": self.camera_name,
        }


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("w", encoding="utf-8")

    def write(self, record: StepRecord) -> None:
        self._fh.write(json.dumps(record.to_json_dict(), ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VideoSink:
    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        release = getattr(self._writer, "release", None)
        if callable(release):
            release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _clamp_index(index: int, size: int) -> int:
    if size <= 0:
        raise ValueError("records must not be empty")
    return max(0, min(index, size - 1))


def _marker_color(record: StepRecord) -> tuple[int, int, int]:
    if record.progress_event == "complete" or record.completed:
        return (80, 220, 80)
    if record.progress_event == "stall":
        return (0, 180, 255)
    if record.progress_event == "regression":
        return (60, 60, 255)
    if record.replanner_called:
        if record.replanner_action == "complete":
            return (255, 220, 80)
        if record.replanner_action == "continue":
            return (255, 170, 0)
        if record.replanner_action == "error":
            return (255, 80, 255)
    return (160, 220, 160)


def _write_status_lines(
    canvas: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    width: int,
    record: StepRecord,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (245, 245, 245)
    small_color = (210, 210, 210)
    y = origin_y + 28
    line_step = 28

    def draw(label: str, value: str, *, scale: float = 0.6, thickness: int = 1, use_small: bool = False) -> None:
        nonlocal y
        text = f"{label}: {value}"
        max_chars = max(12, int(width / 12))
        for line in textwrap.wrap(text, width=max_chars) or [""]:
            cv2.putText(
                canvas,
                line,
                (origin_x + 12, y),
                font,
                scale,
                small_color if use_small else color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_step

    draw("prompt", record.prompt)
    draw("step", str(record.step))
    draw("progress", f"{record.progress:.3f}")
    draw("event", record.progress_event)
    draw("trigger", record.trigger_reason or "-", scale=0.52, use_small=True)
    replanner_value = "called"
    if record.replanner_called:
        replanner_value = f"{record.replanner_action or '-'} | {record.replanner_reason or '-'}"
    else:
        replanner_value = "not called"
    draw("replanner", replanner_value, scale=0.52, use_small=True)


def compose_frame(
    frame: np.ndarray,
    records: Sequence[StepRecord],
    current_index: int,
    complete_threshold: float,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape (H, W, 3)")

    if not records:
        raise ValueError("records must not be empty")

    current_index = _clamp_index(current_index, len(records))
    current = records[current_index]

    frame_h, frame_w = frame.shape[:2]
    status_width = max(320, frame_w // 2)
    progress_height = max(170, frame_h // 2)
    canvas_h = frame_h + progress_height
    canvas_w = frame_w + status_width

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=frame.dtype)
    canvas[:frame_h, :frame_w] = frame
    if status_width > 0:
        canvas[:frame_h, frame_w:] = (22, 22, 22)
    if progress_height > 0:
        canvas[frame_h:, :] = (14, 14, 14)

    if frame_w > 2 and frame_h > 2:
        cv2.rectangle(canvas, (1, 1), (frame_w - 2, frame_h - 2), (60, 60, 60), 1)
    cv2.rectangle(canvas, (frame_w, 0), (canvas_w - 1, frame_h - 1), (70, 70, 70), 1)
    cv2.rectangle(canvas, (0, frame_h), (canvas_w - 1, canvas_h - 1), (70, 70, 70), 1)

    _write_status_lines(
        canvas,
        origin_x=frame_w,
        origin_y=0,
        width=status_width,
        record=current,
    )

    plot_x0 = 18
    plot_y0 = frame_h + 24
    plot_w = canvas_w - 36
    plot_h = progress_height - 48
    plot_x1 = plot_x0 + plot_w
    plot_y1 = plot_y0 + plot_h

    cv2.putText(
        canvas,
        "progress history",
        (plot_x0, frame_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )

    threshold_y = plot_y0 + int(round((1.0 - complete_threshold) * max(plot_h - 1, 1)))
    cv2.line(canvas, (plot_x0, threshold_y), (plot_x1, threshold_y), (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"complete >= {complete_threshold:.2f}",
        (plot_x0 + 6, max(plot_y0 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 220),
        1,
        cv2.LINE_AA,
    )

    history = list(records[: current_index + 1])
    if len(history) == 1:
        points = [(plot_x0 + plot_w // 2, plot_y0 + int((1.0 - history[0].progress) * max(plot_h - 1, 1)))]
    else:
        points = []
        for idx, record in enumerate(history):
            x = plot_x0 + int(round(idx * plot_w / max(len(history) - 1, 1)))
            y = plot_y0 + int(round((1.0 - record.progress) * max(plot_h - 1, 1)))
            points.append((x, y))

    for p0, p1 in zip(points, points[1:]):
        cv2.line(canvas, p0, p1, (90, 180, 90), 2, cv2.LINE_AA)

    for idx, record in enumerate(history):
        point = points[idx]
        cv2.circle(canvas, point, 4, _marker_color(record), -1, cv2.LINE_AA)

    cv2.circle(canvas, points[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (100, 100, 100), 1)

    return canvas


@dataclass(frozen=True)
class ProgressReadout:
    scalar: float
    horizon: tuple[float, ...]
    mean: float
    last: float


def _extract_progress(output: Any) -> ProgressReadout | None:
    if isinstance(output, dict):
        progress = output.get("progress")
    else:
        progress = getattr(output, "progress", None)
    if progress is None:
        return None
    progress_array = np.asarray(progress).astype(np.float32).reshape(-1)
    if progress_array.size == 0:
        return None
    horizon = tuple(float(v) for v in progress_array.tolist())
    return ProgressReadout(
        scalar=float(progress_array[0]),
        horizon=horizon,
        mean=float(progress_array.mean()),
        last=float(progress_array[-1]),
    )


def _maybe_write_frame(
    env: Any,
    camera_name: str,
    step_idx: int,
    write_video_frame: Callable[[Any], None] | None,
) -> None:
    if write_video_frame is None:
        return
    frame = None
    get_image = getattr(env, "get_image", None)
    if callable(get_image):
        frame = get_image(camera_name, step_idx)
    write_video_frame(frame)


def open_video_writer(path: Path, *, width: int, height: int, fps: float) -> VideoSink:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return VideoSink(writer)


def run_replay_visualization(
    *,
    env: Any,
    policy: _ReplayRunPolicyProtocol,
    planner: Any,
    config: ReplayRunConfig,
    write_video_frame: Callable[[Any], None] | None,
    write_record: Callable[[StepRecord], None] | None,
) -> list[StepRecord]:
    tracker = ReplayTaskProgressTracker(
        complete_threshold=config.complete_threshold,
        stall_threshold=config.stall_threshold,
        stall_steps=config.stall_steps,
        regression_threshold=config.regression_threshold,
    )

    if hasattr(env, "set_cursor"):
        env.set_cursor(config.start_step)

    records: list[StepRecord] = []
    current_policy_prompt = config.prompt
    prompt_history: list[dict[str, object]] = []

    first_infer_done = False
    episode_length = int(getattr(env, "num_steps", 0) or 0)
    while True:
        step_idx = env.get_cursor()
        if config.end_step is not None and step_idx >= config.end_step:
            break
        if hasattr(env, "is_episode_complete") and env.is_episode_complete():
            break

        observation = env.get_observation()
        if not first_infer_done:
            _log(f"running first policy.infer at step {step_idx} (JIT compile pass; may take minutes)")
        t0 = time.monotonic()
        output = policy.infer(observation)
        dt = time.monotonic() - t0
        readout = _extract_progress(output)
        if readout is None:
            raise ValueError("policy output did not include progress")
        if not first_infer_done:
            _log(
                f"first policy.infer done in {dt:.1f}s  "
                f"horizon_len={len(readout.horizon)}  "
                f"horizon[0]={readout.scalar:.4f} mean={readout.mean:.4f} last={readout.last:.4f}"
            )
            first_infer_done = True
        elif step_idx % 20 == 0:
            expected_dbg = (
                step_idx / (episode_length - 1) if episode_length > 1 else None
            )
            _log(
                f"step {step_idx} infer={dt*1000:.0f}ms  "
                f"progress[0]={readout.scalar:.3f} mean={readout.mean:.3f} last={readout.last:.3f}"
                + (f"  expected≈{expected_dbg:.3f}" if expected_dbg is not None else "")
            )
        progress = readout.scalar
        expected_progress = (
            step_idx / (episode_length - 1) if episode_length > 1 else None
        )

        decision = tracker.observe(progress)
        trigger_reason = decision.detail
        replanner_called = False
        replanner_action = ""
        replanner_reason = ""
        completed = decision.event == "complete"

        should_replan = config.enable_replanner and (
            decision.event in {"stall", "regression"}
            or (decision.event == "complete" and config.progress_confirm_with_replanner)
        )
        if should_replan:
            replanner_called = True
            _log(f"step {step_idx} calling planner.plan (event={decision.event}, progress={progress:.3f})")
            try:
                replanner_decision = planner.plan(
                    task_prompt=config.prompt,
                    current_policy_prompt=current_policy_prompt,
                    executed_policy_steps=len(records) + 1,
                    prompt_history=prompt_history,
                )
            except Exception as exc:  # noqa: BLE001
                replanner_action = "error"
                replanner_reason = str(exc)
                if trigger_reason:
                    trigger_reason = f"{trigger_reason}; replanner failed: {exc}"
                else:
                    trigger_reason = f"replanner failed: {exc}"
            else:
                replanner_action = str(getattr(replanner_decision, "action", "")).strip()
                replanner_reason = str(getattr(replanner_decision, "reason", "")).strip()
                if replanner_action == "continue":
                    next_prompt = str(getattr(replanner_decision, "prompt", "")).strip()
                    if next_prompt:
                        current_policy_prompt = next_prompt
                    prompt_history.append(
                        {
                            "policy_steps": len(records) + 1,
                            "prompt": current_policy_prompt,
                            "reason": replanner_reason,
                        }
                    )
                    completed = False
                    if decision.event == "complete":
                        trigger_reason = trigger_reason or "progress threshold reached"
                        trigger_reason = f"{trigger_reason}; replanner requested continue"
                    else:
                        trigger_reason = trigger_reason or f"progress {decision.event}"
                elif replanner_action == "complete":
                    prompt_history.append(
                        {
                            "policy_steps": len(records) + 1,
                            "prompt": current_policy_prompt,
                            "reason": replanner_reason or trigger_reason,
                        }
                    )
                    completed = True
                    if not trigger_reason:
                        trigger_reason = "progress threshold reached"
                else:
                    prompt_history.append(
                        {
                            "policy_steps": len(records) + 1,
                            "prompt": current_policy_prompt,
                            "reason": replanner_reason or trigger_reason,
                        }
                    )

        record = StepRecord(
            step=step_idx,
            prompt=current_policy_prompt,
            progress=progress,
            progress_horizon=readout.horizon,
            progress_horizon_mean=readout.mean,
            progress_horizon_last=readout.last,
            expected_progress=expected_progress,
            progress_event=decision.event,
            trigger_reason=trigger_reason,
            replanner_called=replanner_called,
            replanner_action=replanner_action,
            replanner_reason=replanner_reason,
            completed=completed,
            camera_name=config.camera_name,
        )
        records.append(record)
        if write_record is not None:
            write_record(record)
        _maybe_write_frame(env, config.camera_name, step_idx, write_video_frame)

        if completed:
            break

    return records


def render_replay_visualization(
    runtime: ReplayVisualizationRuntime,
    *,
    output_video: Path,
    dump_jsonl: Path | None = None,
    fps: float = 25.0,
) -> list[StepRecord]:
    jsonl_recorder = JsonlRecorder(dump_jsonl) if dump_jsonl is not None else None
    try:
        records = run_replay_visualization(
            env=runtime.env,
            policy=runtime.policy,
            planner=runtime.planner,
            config=runtime.run_config,
            write_video_frame=None,
            write_record=jsonl_recorder.write if jsonl_recorder is not None else None,
        )
        if not records:
            return records

        get_image = getattr(runtime.env, "get_image", None)
        if not callable(get_image):
            raise AttributeError("environment does not provide get_image")

        camera_name = runtime.run_config.camera_name
        first_frame = get_image(camera_name, records[0].step)
        if first_frame is None:
            raise ValueError("environment returned no frame")
        first_canvas = compose_frame(
            first_frame,
            records,
            0,
            runtime.run_config.complete_threshold,
        )

        video_sink = open_video_writer(
            output_video,
            width=int(first_canvas.shape[1]),
            height=int(first_canvas.shape[0]),
            fps=fps,
        )
        try:
            video_sink.write(first_canvas)
            for index, record in enumerate(records[1:], start=1):
                frame = get_image(camera_name, record.step)
                if frame is None:
                    raise ValueError("environment returned no frame")
                video_sink.write(
                    compose_frame(
                        frame,
                        records,
                        index,
                        runtime.run_config.complete_threshold,
                    )
                )
        finally:
            video_sink.close()
        return records
    finally:
        if jsonl_recorder is not None:
            jsonl_recorder.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Path to the replay dataset.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the local checkpoint directory.")
    parser.add_argument(
        "--output-video",
        default=None,
        help=f"Path to the output MP4 video. Default: {DEFAULT_OUTPUT_DIR}/<dataset_stem>_progress.mp4",
    )
    parser.add_argument("--prompt", required=True, help="Manipulation prompt to visualize.")
    parser.add_argument("--start-step", type=int, default=None, help="First replay step to visualize.")
    parser.add_argument("--end-step", type=int, default=None, help="Last replay step to visualize.")
    parser.add_argument(
        "--dump-jsonl",
        default=None,
        help=(
            "JSONL trace output path. Pass '' / 'none' to disable. "
            f"Default: {DEFAULT_OUTPUT_DIR}/<dataset_stem>_progress.jsonl"
        ),
    )
    parser.add_argument(
        "--camera-name",
        default="cam_high",
        help="Replay camera to render. Default: cam_high",
    )
    parser.add_argument(
        "--task-decompose",
        action="store_true",
        help="Reserved flag for future multi-subtask replay support.",
    )
    parser.add_argument(
        "--no-replanner",
        action="store_true",
        help="Skip LLM replanner entirely (diagnose raw progress head only).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Args:
    ns = build_parser().parse_args(argv)
    output_video = ns.output_video or _default_output_video(ns.dataset_path)
    if ns.dump_jsonl is None:
        dump_jsonl: str | None = _default_dump_jsonl(ns.dataset_path)
    elif ns.dump_jsonl.strip().lower() in {"", "none", "off", "false"}:
        dump_jsonl = None
    else:
        dump_jsonl = ns.dump_jsonl
    return Args(
        dataset_path=ns.dataset_path,
        checkpoint_dir=ns.checkpoint_dir,
        output_video=output_video,
        prompt=ns.prompt,
        start_step=ns.start_step,
        end_step=ns.end_step,
        dump_jsonl=dump_jsonl,
        camera_name=ns.camera_name,
        task_decompose=ns.task_decompose,
        no_replanner=ns.no_replanner,
    )


def _load_progress_metadata(checkpoint_dir: Path) -> dict[str, object]:
    metadata_path = checkpoint_dir / "assets" / "progress_metadata.json"
    if not metadata_path.exists():
        return {"has_progress_head": False}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _candidate_train_config_names(checkpoint_dir: Path) -> list[str]:
    candidates: list[str] = []

    def _add(name: str | None) -> None:
        if not name:
            return
        value = name.strip()
        if not value or value in candidates:
            return
        candidates.append(value)

    candidate_files = [
        checkpoint_dir / "assets" / "train_config.json",
        checkpoint_dir / "assets" / "checkpoint_config.json",
        checkpoint_dir / "train_config.json",
        checkpoint_dir / "checkpoint_config.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _add(candidate.read_text(encoding="utf-8"))
            continue
        if isinstance(payload, dict):
            for key in ("config", "config_name", "name"):
                value = payload.get(key)
                if isinstance(value, str):
                    _add(value)
        elif isinstance(payload, str):
            _add(payload)

    assets_dir = checkpoint_dir / "assets"
    if assets_dir.is_dir():
        for child in sorted(assets_dir.iterdir()):
            if child.is_dir():
                _add(child.name)

    skip = {"checkpoints", "checkpoint", "assets"}
    path_parts = [
        checkpoint_dir.parent.parent.name,
        checkpoint_dir.parent.name,
        checkpoint_dir.name,
    ]
    for part in path_parts:
        if part and part not in skip and not part.isdigit():
            _add(part)

    _add("debug_pi05")
    return candidates


def _load_train_config(checkpoint_dir: Path) -> Any:
    candidates = _candidate_train_config_names(checkpoint_dir)
    last_error: Exception | None = None
    for name in candidates:
        try:
            return _config.get_config(name)
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Could not resolve a train config for checkpoint '{checkpoint_dir}'. "
        f"Tried: {candidates}. Last error: {last_error}"
    )


def build_runtime(config: ReplayProgressVisualizationConfig | Args) -> ReplayVisualizationRuntime:
    if isinstance(config, Args):
        config = build_config(config)

    checkpoint_dir = config.checkpoint_dir
    _log(f"resolving train config from {checkpoint_dir}")
    train_config = _load_train_config(checkpoint_dir)
    _log(f"train config resolved: {getattr(train_config, 'name', train_config)}")
    _log(f"opening replay dataset {config.dataset_path}")
    env = ReplayEnvironment(
        dataset_path=str(config.dataset_path),
        prompt=config.prompt,
        max_steps=config.end_step,
    )
    _log("replay dataset opened")
    _log("initializing planner config (PlannerConfig)")
    planner_config = PlannerConfig()
    if config.no_replanner:
        _log("--no-replanner: skipping service-config validation and planner init")
        planner_config.validate_motion_limits()
    else:
        planner_config.validate()
    _log("loading checkpoint params (this can take minutes on first JIT compile)")
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=config.prompt,
    )
    _log("policy loaded")
    if config.no_replanner:
        planner = None
        _log("planner disabled")
    else:
        planner = ReplayManipulationPromptPlanner(env, planner_config)
        _log("planner initialized")
    run_config = ReplayRunConfig(
        prompt=config.prompt,
        start_step=config.start_step or 0,
        end_step=config.end_step,
        camera_name=config.camera_name,
        complete_threshold=planner_config.progress_complete_threshold,
        stall_threshold=planner_config.progress_stall_threshold,
        stall_steps=planner_config.progress_stall_steps,
        regression_threshold=planner_config.progress_regression_threshold,
        progress_confirm_with_replanner=planner_config.progress_confirm_with_replanner,
        enable_replanner=not config.no_replanner,
    )
    return ReplayVisualizationRuntime(
        env=env,
        policy=policy,
        planner=planner,
        planner_config=planner_config,
        run_config=run_config,
    )


def validate_args(args: Args) -> None:
    dataset_path = Path(args.dataset_path)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not dataset_path.exists():
        raise ValueError(f"dataset_path does not exist: {dataset_path}")
    if not checkpoint_dir.exists():
        raise ValueError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if args.task_decompose:
        raise NotImplementedError("--task-decompose is reserved for a future iteration")
    if args.end_step is not None and args.start_step is not None and args.start_step > args.end_step:
        raise ValueError("start_step must be <= end_step")

    metadata = _load_progress_metadata(checkpoint_dir)
    if not bool(metadata.get("has_progress_head", False)):
        raise ValueError("visualizer requires a progress-head checkpoint")


def build_config(args: Args) -> ReplayProgressVisualizationConfig:
    validate_args(args)
    return ReplayProgressVisualizationConfig(
        dataset_path=Path(args.dataset_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_video=Path(args.output_video),
        prompt=args.prompt,
        start_step=args.start_step,
        end_step=args.end_step,
        dump_jsonl=Path(args.dump_jsonl) if args.dump_jsonl else None,
        camera_name=args.camera_name,
        task_decompose=args.task_decompose,
        no_replanner=args.no_replanner,
    )


def main(argv: Sequence[str] | Args | None = None) -> int:
    args = argv if isinstance(argv, Args) else parse_args(argv)
    config = build_config(args)
    runtime = build_runtime(config)
    try:
        render_replay_visualization(
            runtime,
            output_video=config.output_video,
            dump_jsonl=config.dump_jsonl,
        )
    finally:
        close = getattr(runtime.env, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
