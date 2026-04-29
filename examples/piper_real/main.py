# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import json
import logging
import os
from pathlib import Path
import re
import textwrap
import time

import cv2
import numpy as np

import tyro

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime.agents import policy_agent as _policy_agent
from examples.piper_real.planner_config import PlannerConfig


DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_REPLAY_MANIPULATE_MAX_STEPS = 64
DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS = 16
DEFAULT_ROBOT_BASE_TOPIC = "/odom_raw"
DEFAULT_ROBOT_BASE_CMD_TOPIC = "/cmd_vel"
_VALID_REPLAY_MODES = {"policy", "planner", "hybrid"}


def _restore_cli_logging() -> None:
    """Keep Python logging visible after ROS node initialization."""

    logging.basicConfig(level=logging.INFO, force=True)


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"  # H100
    port: int = 9000
    action_horizon: int = 16
    num_episodes: int = 1
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    save_log: bool = False
    prompt: str = ""
    replay_dataset: str = ""  # Path to HDF5 episode file for offline replay
    replay_mode: str = "policy"  # policy | planner | hybrid
    replay_manipulate_max_steps: int = DEFAULT_REPLAY_MANIPULATE_MAX_STEPS
    replay_manipulate_replan_interval_steps: int = (
        DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS
    )
    progress_head_mode: str = "auto"  # auto | force | off
    use_llm_planner: bool = False
    use_robot_base: bool = False
    robot_base_topic: str = DEFAULT_ROBOT_BASE_TOPIC
    robot_base_cmd_topic: str = DEFAULT_ROBOT_BASE_CMD_TOPIC
    navigation_only: bool = False  # Run navigation only, skip manipulation
    skip_server_checks: bool = False
    server_check_timeout_sec: float = 5.0
    visualize: bool = (
        False  # Show camera views + subtask overlay during replay (web UI)
    )
    visualize_port: int = 7860  # Port for the visualization web server
    save_path: str = ""  # Optional MP4 output path synchronized with visualize updates
    replay_debug_export_dir: str = (
        ""  # Optional directory for hit-cap prompt/frame debug exports
    )
    visualize_playback_rate: float = (
        1.0  # Relative replay visualization speed (1.0 = dataset FPS)
    )
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)


def _create_policy_agent(args: Args) -> _policy_agent.PolicyAgent:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )
    )
    agent.policy_metadata = metadata
    return agent


def _run_required_server_checks(
    args: Args,
    *,
    needs_pi0: bool = False,
    needs_planner: bool = False,
) -> bool:
    if args.skip_server_checks or (not needs_pi0 and not needs_planner):
        return True

    from examples.piper_real import server_checks as _server_checks

    try:
        if needs_planner:
            _server_checks.check_planner_server(
                args.planner.base_url,
                expected_model=args.planner.model,
                timeout_sec=args.server_check_timeout_sec,
                api_key=args.planner.api_key or None,
            )
        if needs_pi0:
            _server_checks.check_pi0_server(
                args.host,
                args.port,
                timeout_sec=args.server_check_timeout_sec,
            )
    except _server_checks.ServerCheckError as exc:
        logging.error("%s", exc)
        return False

    return True


def _log_replay_summary(environment, executed_steps: int) -> None:
    if not environment.predicted_actions:
        logging.warning("Replay finished without any predicted actions.")
        return

    predicted = np.stack(environment.predicted_actions)
    gt_indices = np.asarray(environment.predicted_action_steps, dtype=np.int64)
    gt = environment.ground_truth_actions[gt_indices]

    arm_dim = min(14, predicted.shape[-1], gt.shape[-1])
    arm_mae = float(np.mean(np.abs(predicted[:, :arm_dim] - gt[:, :arm_dim])))

    base_suffix = ", base_mae=N/A"
    if predicted.shape[-1] >= 16:
        if environment.ground_truth_base_actions is not None:
            gt_base = environment.ground_truth_base_actions[gt_indices]
            base_mae = float(np.mean(np.abs(predicted[:, 14:16] - gt_base)))
            base_suffix = f", base_mae={base_mae:.6f}"
        elif gt.shape[-1] >= 16:
            gt_base = gt[:, 14:16]
            base_mae = float(np.mean(np.abs(predicted[:, 14:16] - gt_base)))
            base_suffix = f", base_mae={base_mae:.6f}"

    logging.info(
        "Replay finished: executed_steps=%d/%d, predicted_steps=%d, arm_mae=%.6f%s",
        executed_steps,
        environment.num_steps,
        len(predicted),
        arm_mae,
        base_suffix,
    )


def _resolve_replay_mode(args: Args) -> str:
    replay_mode = args.replay_mode.strip().lower()
    if replay_mode not in _VALID_REPLAY_MODES:
        raise ValueError(
            f"--replay-mode must be one of {sorted(_VALID_REPLAY_MODES)}, got {args.replay_mode!r}."
        )
    if args.use_llm_planner and replay_mode == "policy":
        logging.info(
            "Replay mode inferred as planner from legacy --use-llm-planner flag."
        )
        return "planner"
    return replay_mode


def _build_ordered_task_memory_runtime(args: Args, environment):
    task_spec_path = args.planner.task_spec_path.strip()
    if not task_spec_path:
        return None

    from examples.piper_real import replay_task_memory as _replay_task_memory

    try:
        runtime = _replay_task_memory.ReplayOrderedTaskMemoryRuntime(
            environment,
            args.planner,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error(
            "Could not initialize ordered task memory from %s: %s",
            task_spec_path,
            exc,
        )
        return None

    logging.info("Ordered task memory enabled: %s", task_spec_path)
    return runtime


def _mark_ordered_task_completed(
    ordered_task_memory_runtime,
    subtask_index: int,
    *,
    reason: str,
) -> None:
    if ordered_task_memory_runtime is None:
        return
    marker = getattr(ordered_task_memory_runtime, "mark_completed_through", None)
    if not callable(marker):
        return
    try:
        marker(subtask_index, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Could not advance ordered task memory after %s: %s",
            reason,
            exc,
        )


def _refresh_environment_observation_cache(environment, *, context: str) -> None:
    refresher = getattr(environment, "refresh_observation_cache", None)
    if not callable(refresher):
        return
    try:
        refresher()
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Could not refresh observation cache before %s; continuing with cached frame: %s",
            context,
            exc,
        )


def _build_replay_subtask_list(args: Args, prompt: str, ordered_task_memory_runtime=None):
    from examples.piper_real import task_decomposer as _task_decomposer

    decomposer = _task_decomposer.TaskDecomposer(args.planner)
    decompose_kwargs: dict[str, str] = {}
    if ordered_task_memory_runtime is not None:
        try:
            decompose_kwargs = ordered_task_memory_runtime.build_context()
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Replay decomposition is proceeding without ordered task memory context: %s",
                exc,
            )
            decompose_kwargs = {}

    try:
        if decompose_kwargs:
            subtask_list = decomposer.decompose(prompt, **decompose_kwargs)
        else:
            subtask_list = decomposer.decompose(prompt)
    except _task_decomposer.DecompositionError as exc:
        if args.navigation_only:
            logging.warning(
                "Replay decomposition failed in navigation-only mode (%s); "
                "using the original prompt as a single navigate subtask.",
                exc,
            )
            return [_task_decomposer.Subtask(type="navigate", prompt=prompt)]
        logging.error("Task decomposition failed: %s", exc)
        return None

    if args.navigation_only and not any(
        subtask.type == "navigate" for subtask in subtask_list
    ):
        logging.warning(
            "Replay decomposition returned no navigate subtasks; "
            "using the original prompt as a single navigate subtask."
        )
        return [_task_decomposer.Subtask(type="navigate", prompt=prompt)]

    return subtask_list


def _resolve_replay_debug_export_dir(args: Args) -> str:
    explicit_dir = args.replay_debug_export_dir.strip()
    if explicit_dir:
        return explicit_dir

    if args.save_path.strip():
        save_path = Path(args.save_path)
        stem_path = save_path.with_suffix("")
        return str(stem_path.parent / f"{stem_path.name}_debug")

    return os.path.join("outputs", "replay_debug")


def _sanitize_debug_path_component(raw_text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw_text.strip().lower()).strip("_")
    if not slug:
        return "subtask"
    return slug[:max_len]


def _compose_replay_debug_frame(
    environment,
    *,
    frame_idx: int,
    header_lines: list[str],
) -> np.ndarray:
    bgr_frames: list[np.ndarray] = []
    for cam_name in environment.camera_names:
        frame = environment.get_image(cam_name, frame_idx)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        canvas = frame.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            cam_name,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        bgr_frames.append(canvas)

    if not bgr_frames:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    tile_h = max(frame.shape[0] for frame in bgr_frames)
    tile_w = max(frame.shape[1] for frame in bgr_frames)
    normalized: list[np.ndarray] = []
    for frame in bgr_frames:
        if frame.shape[:2] != (tile_h, tile_w):
            frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        normalized.append(frame)

    rows: list[np.ndarray] = []
    cols = 2
    for start in range(0, len(normalized), cols):
        row_frames = normalized[start : start + cols]
        if len(row_frames) < cols:
            row_frames.append(np.zeros_like(normalized[0]))
        rows.append(cv2.hconcat(row_frames))
    sheet = cv2.vconcat(rows)

    wrapped_lines: list[str] = []
    wrap_width = max(32, sheet.shape[1] // 14)
    for line in header_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=wrap_width) or [""])

    line_height = 24
    header_height = max(44, 12 + len(wrapped_lines) * line_height)
    header = np.zeros((header_height, sheet.shape[1], 3), dtype=np.uint8)
    header[:] = (22, 22, 22)
    y = 26
    for line in wrapped_lines:
        cv2.putText(
            header,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return cv2.vconcat([header, sheet])


def _export_replay_manipulation_cap_debug(
    environment,
    *,
    export_base_dir: str,
    subtask_prompt: str,
    current_policy_prompt: str,
    prompt_history: list[dict[str, object]],
    subtask_start_cursor: int,
    executed_steps: int,
    max_steps: int,
    subtask_index: int | None,
    total_subtasks: int | None,
) -> Path:
    export_root = Path(export_base_dir)
    subtask_prefix = (
        f"subtask_{subtask_index:02d}"
        if subtask_index is not None
        else "subtask_unknown"
    )
    export_dir = export_root / f"{subtask_prefix}_{_sanitize_debug_path_component(subtask_prompt)}"
    export_dir.mkdir(parents=True, exist_ok=True)

    history_items = prompt_history[-5:]
    if not history_items:
        history_items = [
            {
                "policy_steps": executed_steps,
                "prompt": current_policy_prompt,
                "reason": "no_prompt_history_available",
            }
        ]

    exported_history: list[dict[str, object]] = []
    for item_idx, item in enumerate(history_items, start=1):
        policy_steps = int(item.get("policy_steps", 0))
        if policy_steps <= 0:
            frame_idx = subtask_start_cursor
        else:
            frame_idx = min(
                subtask_start_cursor + policy_steps - 1,
                max(environment.num_steps - 1, 0),
            )
        image_name = f"prompt_{item_idx:02d}_step_{policy_steps:03d}.png"
        image_path = export_dir / image_name
        prompt_text = str(item.get("prompt", ""))
        reason_text = str(item.get("reason", ""))
        debug_frame = _compose_replay_debug_frame(
            environment,
            frame_idx=frame_idx,
            header_lines=[
                f"Subtask {subtask_index}/{total_subtasks} hit cap at {max_steps} steps",
                f"Representative frame: replay step {frame_idx}",
                f"Policy steps at replan: {policy_steps}",
                f"Prompt: {prompt_text}",
                f"Reason: {reason_text}",
            ],
        )
        cv2.imwrite(str(image_path), debug_frame)
        exported_history.append(
            {
                "policy_steps": policy_steps,
                "prompt": prompt_text,
                "reason": reason_text,
                "frame_idx": frame_idx,
                "image_path": str(image_path),
            }
        )

    final_frame_idx = min(
        max(subtask_start_cursor, environment.get_cursor() - 1),
        max(environment.num_steps - 1, 0),
    )
    final_frame_path = export_dir / "final_frame.png"
    final_frame = _compose_replay_debug_frame(
        environment,
        frame_idx=final_frame_idx,
        header_lines=[
            f"Subtask {subtask_index}/{total_subtasks} final frame at cap",
            f"Replay step {final_frame_idx} | executed_steps={executed_steps} | max_steps={max_steps}",
            f"Current policy prompt: {current_policy_prompt}",
        ],
    )
    cv2.imwrite(str(final_frame_path), final_frame)

    manifest = {
        "subtask_index": subtask_index,
        "total_subtasks": total_subtasks,
        "subtask_prompt": subtask_prompt,
        "subtask_start_cursor": subtask_start_cursor,
        "executed_steps": executed_steps,
        "max_steps": max_steps,
        "final_cursor": environment.get_cursor(),
        "final_frame_idx": final_frame_idx,
        "current_policy_prompt": current_policy_prompt,
        "camera_names": list(environment.camera_names),
        "history_window_size": len(exported_history),
        "recent_prompt_history": exported_history,
        "final_frame_path": str(final_frame_path),
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return export_dir


def _run_manipulation_subtask(
    environment,
    agent: _policy_agent.PolicyAgent,
    manipulation_planner,
    *,
    subtask_prompt: str,
    max_steps: int,
    replan_interval_steps: int,
    progress_complete_threshold: float,
    progress_stall_threshold: float,
    progress_stall_steps: int,
    progress_regression_threshold: float,
    progress_confirm_with_replanner: bool,
    progress_head_mode: str = "auto",
    debug_export_dir: str = "",
    subtask_index: int | None = None,
    total_subtasks: int | None = None,
    visualizer=None,
) -> dict[str, object]:
    prompt_history: list[dict[str, object]] = []
    current_policy_prompt = subtask_prompt
    prompt_queries = 0
    subtask_start_cursor = (
        environment.get_cursor() if hasattr(environment, "get_cursor") else 0
    )
    progress_head_mode = str(progress_head_mode or "auto").strip().lower()
    if progress_head_mode not in {"auto", "force", "off"}:
        raise ValueError("progress_head_mode must be one of: auto, force, off")
    metadata_has_progress_head = bool(
        getattr(agent, "policy_metadata", {}).get("has_progress_head", False)
    )
    has_progress_head = (
        metadata_has_progress_head
        if progress_head_mode == "auto"
        else progress_head_mode == "force"
    )
    progress_tracker = ReplayTaskProgressTracker(
        complete_threshold=progress_complete_threshold,
        stall_threshold=progress_stall_threshold,
        stall_steps=progress_stall_steps,
        regression_threshold=progress_regression_threshold,
    )
    visual_completion_detector = None
    try:
        from examples.piper_real.visual_completion_detector import (
            SandwichVisualCompletionDetector,
        )

        candidate_detector = SandwichVisualCompletionDetector(subtask_prompt)
        if candidate_detector.enabled:
            visual_completion_detector = candidate_detector
            logging.info(
                "Visual completion detector enabled for manipulate subtask %s/%s: %s",
                subtask_index if subtask_index is not None else "?",
                total_subtasks if total_subtasks is not None else "?",
                subtask_prompt,
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Visual completion detector unavailable: %s", exc)
    logging.info(
        "Manipulate subtask %s/%s start: prompt=%s max_steps=%d "
        "replan_interval_steps=%d has_progress_head=%s progress_head_mode=%s",
        subtask_index if subtask_index is not None else "?",
        total_subtasks if total_subtasks is not None else "?",
        subtask_prompt,
        max_steps,
        replan_interval_steps,
        has_progress_head,
        progress_head_mode,
    )

    def _replan(policy_steps: int) -> bool:
        nonlocal current_policy_prompt
        nonlocal prompt_queries
        logging.info(
            "Replay manipulate replanner query at %d policy steps for: %s",
            policy_steps,
            subtask_prompt,
        )
        try:
            decision = manipulation_planner.plan(
                task_prompt=subtask_prompt,
                current_policy_prompt=current_policy_prompt,
                executed_policy_steps=policy_steps,
                prompt_history=prompt_history,
            )
            prompt_queries += 1
        except Exception as exc:  # noqa: BLE001
            environment.set_prompt(current_policy_prompt)
            agent.reset()
            prompt_history.append(
                {
                    "policy_steps": policy_steps,
                    "prompt": current_policy_prompt,
                    "reason": f"replan_failed: {exc}",
                }
            )
            logging.error(
                "Replay manipulate replanner failed after %d policy steps; reusing current prompt: %s",
                policy_steps,
                exc,
            )
            return True

        if decision.action == "complete":
            logging.info(
                "Replay manipulate replanner marked subtask complete after %d policy steps: %s",
                policy_steps,
                decision.reason,
            )
            return False

        current_policy_prompt = decision.prompt
        prompt_history.append(
            {
                "policy_steps": policy_steps,
                "prompt": decision.prompt,
                "reason": decision.reason,
            }
        )
        environment.set_prompt(decision.prompt)
        agent.reset()
        logging.info(
            "Replay manipulate replanner prompt update after %d policy steps: %s",
            policy_steps,
            decision.prompt,
        )
        return True

    if not has_progress_head:
        if not _replan(0):
            return {
                "executed_steps": 0,
                "prompt_queries": prompt_queries,
                "completed": True,
                "completed_by_replan": True,
                "completed_by_progress": False,
                "last_policy_prompt": current_policy_prompt,
                "stop_reason": "replanner_complete",
            }
    else:
        logging.info("Replay manipulate subtask is using progress-first mode with replanner fallback.")

    executed_steps = 0
    aborted_by_user = False
    while executed_steps < max_steps and not environment.is_episode_complete():
        observation = environment.get_observation()
        action = agent.get_action(observation)
        environment.apply_action(action)
        executed_steps += 1

        progress_value = action.get("progress")
        if visualizer is not None:
            step_idx = environment.get_cursor() - 1
            if not visualizer.update(
                step_idx, extra_info=f"policy prompt: {current_policy_prompt}"
            ):
                logging.info(
                    "Replay manipulate aborted by user at policy step %d.",
                    executed_steps,
                )
                aborted_by_user = True
                break

        if (
            visual_completion_detector is not None
            and executed_steps % replan_interval_steps == 0
        ):
            try:
                visual_decision = visual_completion_detector.observe(environment)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Visual completion detector failed after %d policy steps for %s: %s",
                    executed_steps,
                    subtask_prompt,
                    exc,
                )
            else:
                logging.info(
                    "Visual completion detector at %d policy steps for %s: complete=%s reason=%s metrics=%s",
                    executed_steps,
                    subtask_prompt,
                    visual_decision.complete,
                    visual_decision.reason,
                    json.dumps(visual_decision.metrics, sort_keys=True),
                )
                if visual_decision.complete:
                    logging.info(
                        "Visual completion detector marked subtask complete after %d policy steps: %s",
                        executed_steps,
                        visual_decision.reason,
                    )
                    return {
                        "executed_steps": executed_steps,
                        "prompt_queries": prompt_queries,
                        "completed": True,
                        "completed_by_replan": False,
                        "completed_by_progress": False,
                        "completed_by_visual": True,
                        "last_policy_prompt": current_policy_prompt,
                        "stop_reason": "visual_complete",
                    }

        if has_progress_head and progress_value is not None:
            decision = progress_tracker.observe(float(progress_value))
            logging.info(
                "Replay manipulate progress step %d: progress=%.4f event=%s",
                executed_steps,
                decision.progress,
                decision.event,
            )
            if decision.event == "complete":
                if not progress_confirm_with_replanner:
                    return {
                        "executed_steps": executed_steps,
                        "prompt_queries": prompt_queries,
                        "completed": True,
                        "completed_by_replan": False,
                        "completed_by_progress": True,
                        "last_policy_prompt": current_policy_prompt,
                        "stop_reason": "progress_complete",
                    }
                if not _replan(executed_steps):
                    return {
                        "executed_steps": executed_steps,
                        "prompt_queries": prompt_queries,
                        "completed": True,
                        "completed_by_replan": True,
                        "completed_by_progress": False,
                        "last_policy_prompt": current_policy_prompt,
                        "stop_reason": "replanner_complete",
                    }
            elif decision.event in {"stall", "regression"}:
                logging.warning(
                    "Replay manipulate progress %s after %d policy steps at %.4f; falling back to replanner.",
                    decision.event,
                    executed_steps,
                    decision.progress,
                )
                if not _replan(executed_steps):
                    return {
                        "executed_steps": executed_steps,
                        "prompt_queries": prompt_queries,
                        "completed": True,
                        "completed_by_replan": True,
                        "completed_by_progress": False,
                        "last_policy_prompt": current_policy_prompt,
                        "stop_reason": "replanner_complete",
                    }
                continue

        if (
            not has_progress_head
            and executed_steps < max_steps
            and not environment.is_episode_complete()
            and executed_steps % replan_interval_steps == 0
            and not _replan(executed_steps)
        ):
            return {
                "executed_steps": executed_steps,
                "prompt_queries": prompt_queries,
                "completed": True,
                "completed_by_replan": True,
                "completed_by_progress": False,
                "last_policy_prompt": current_policy_prompt,
                "stop_reason": "replanner_complete",
            }

        if (
            executed_steps < max_steps
            and not environment.is_episode_complete()
            and has_progress_head
            and progress_value is None
            and executed_steps % replan_interval_steps == 0
            and not _replan(executed_steps)
        ):
            return {
                "executed_steps": executed_steps,
                "prompt_queries": prompt_queries,
                "completed": True,
                "completed_by_replan": True,
                "completed_by_progress": False,
                "last_policy_prompt": current_policy_prompt,
                "stop_reason": "replanner_complete",
            }

    if executed_steps >= max_steps and not environment.is_episode_complete():
        logging.info(
            "Replay manipulate subtask hit fixed cap at %d policy steps.", max_steps
        )
        if debug_export_dir:
            try:
                export_dir = _export_replay_manipulation_cap_debug(
                    environment,
                    export_base_dir=debug_export_dir,
                    subtask_prompt=subtask_prompt,
                    current_policy_prompt=current_policy_prompt,
                    prompt_history=prompt_history,
                    subtask_start_cursor=subtask_start_cursor,
                    executed_steps=executed_steps,
                    max_steps=max_steps,
                    subtask_index=subtask_index,
                    total_subtasks=total_subtasks,
                )
                logging.info(
                    "Replay manipulate cap debug exported: %s",
                    export_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "Replay manipulate cap debug export failed: %s",
                    exc,
                )
    elif environment.is_episode_complete():
        logging.info(
            "Replay manipulate subtask stopped because the replay dataset is exhausted."
        )

    if aborted_by_user:
        stop_reason = "user_abort"
    elif executed_steps >= max_steps and not environment.is_episode_complete():
        stop_reason = "step_cap"
    elif environment.is_episode_complete():
        stop_reason = "replay_exhausted"
    else:
        stop_reason = "incomplete"

    return {
        "executed_steps": executed_steps,
        "prompt_queries": prompt_queries,
        "completed": False,
        "completed_by_replan": False,
        "completed_by_progress": False,
        "last_policy_prompt": current_policy_prompt,
        "stop_reason": stop_reason,
    }


def _log_hybrid_replay_summary(
    environment,
    *,
    total_subtasks: int,
    navigate_subtasks: int,
    manipulate_subtasks: int,
    policy_steps: int,
    prompt_queries: int,
) -> None:
    logging.info(
        "Hybrid replay completed: subtasks=%d, navigate=%d, manipulate=%d, "
        "policy_steps=%d, prompt_queries=%d, replay_cursor=%d/%d",
        total_subtasks,
        navigate_subtasks,
        manipulate_subtasks,
        policy_steps,
        prompt_queries,
        environment.get_cursor(),
        environment.num_steps,
    )
    if policy_steps > 0:
        _log_replay_summary(environment, policy_steps)


def _get_visualizer_step(environment) -> int:
    if environment.num_steps <= 0:
        return 0
    return min(max(environment.get_cursor(), 0), environment.num_steps - 1)


def _build_navigation_visualizer_updater(
    environment, visualizer, *, playback_rate: float, enabled: bool
):
    frame_interval_sec = 0.0
    if enabled and environment.fps > 0:
        frame_interval_sec = 1.0 / (environment.fps * playback_rate)

    next_refresh_at: float | None = None

    def _on_nav_step(step_idx: int) -> bool:
        nonlocal next_refresh_at
        if frame_interval_sec > 0:
            now = time.monotonic()
            if next_refresh_at is None:
                next_refresh_at = now
            elif now < next_refresh_at:
                time.sleep(next_refresh_at - now)
                now = time.monotonic()
            next_refresh_at = now + frame_interval_sec

        extra_info = (
            f"navigate timeline frame: {step_idx + 1}/{environment.num_steps}, "
            f"playback: {playback_rate:.2f}x"
        )
        return visualizer.update(step_idx, extra_info=extra_info)

    return _on_nav_step


def _run_replay_inference(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error("--replay-dataset currently supports only --num-episodes=1.")
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay mode.")

    logging.info("Replay mock mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
    )

    if not _run_required_server_checks(args, needs_pi0=True):
        return

    agent = _create_policy_agent(args)

    if (
        args.max_episode_steps == DEFAULT_MAX_EPISODE_STEPS
        and environment.num_steps > args.max_episode_steps
    ):
        logging.warning(
            "Replay dataset has %d steps; default --max-episode-steps=%d will truncate it. "
            "Pass --max-episode-steps 0 to run the full dataset.",
            environment.num_steps,
            args.max_episode_steps,
        )

    visualizer = _replay_visualizer.ReplayVisualizer(
        environment,
        enabled=args.visualize,
        port=args.visualize_port,
        save_path=args.save_path,
    )
    visualizer.set_subtask_context(1, 1, "policy", prompt)

    environment.reset()
    agent.reset()

    executed_steps = 0
    try:
        while not environment.is_episode_complete():
            observation = environment.get_observation()
            action = agent.get_action(observation)
            environment.apply_action(action)
            executed_steps += 1

            if not visualizer.update(executed_steps - 1):
                logging.info("Replay aborted by user at step %d.", executed_steps)
                break

            if args.max_episode_steps > 0 and executed_steps >= args.max_episode_steps:
                logging.info(
                    "Replay stopped early at %d steps due to --max-episode-steps=%d.",
                    executed_steps,
                    args.max_episode_steps,
                )
                break
    finally:
        visualizer.close()

    _log_replay_summary(environment, executed_steps)


def _run_replay_planner(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import replay_planner as _replay_planner
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error(
            "--replay-dataset with --use-llm-planner currently supports only --num-episodes=1."
        )
        return

    if not prompt:
        logging.error(
            "--use-llm-planner with --replay-dataset requires a non-empty --prompt."
        )
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay planner mode.")

    if args.visualize_playback_rate <= 0:
        logging.error("--visualize-playback-rate must be positive.")
        return

    args.planner.validate_service_config()
    args.planner.validate_motion_limits()
    if not _run_required_server_checks(args, needs_planner=True):
        return

    logging.info("Replay planner mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
        max_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    if not args.navigation_only:
        logging.warning(
            "Replay planner mode evaluates navigate subtasks only; manipulate subtasks will be skipped."
        )

    subtask_list = _build_replay_subtask_list(args, prompt)
    if subtask_list is None:
        environment.close()
        return

    visualizer = _replay_visualizer.ReplayVisualizer(
        environment,
        enabled=args.visualize,
        port=args.visualize_port,
        save_path=args.save_path,
    )
    on_nav_step = _build_navigation_visualizer_updater(
        environment,
        visualizer,
        playback_rate=args.visualize_playback_rate,
        enabled=args.visualize,
    )

    planner = _replay_planner.OfflineReplayNavigationPlanner(
        environment,
        args.planner,
        on_step_callback=on_nav_step,
    )
    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing replay subtask %d/%d [%s]: %s",
                idx + 1,
                len(subtask_list),
                subtask.type,
                subtask.prompt,
            )

            if subtask.type == "navigate":
                visualizer.set_subtask_context(
                    idx + 1, len(subtask_list), subtask.type, subtask.prompt
                )
                if not on_nav_step(_get_visualizer_step(environment)):
                    logging.info(
                        "Replay planner aborted by user before subtask %d/%d.",
                        idx + 1,
                        len(subtask_list),
                    )
                    return
                if not planner.run(task_prompt=subtask.prompt):
                    logging.error(
                        "Replay navigation failed at subtask %d/%d; aborting.",
                        idx + 1,
                        len(subtask_list),
                    )
                    return
                logging.info(
                    "Replay navigate subtask %d/%d succeeded at replay step %d/%d.",
                    idx + 1,
                    len(subtask_list),
                    planner.current_step,
                    environment.num_steps,
                )
            elif args.navigation_only:
                logging.info("Manipulate (skipped): %s", subtask.prompt)
            else:
                logging.info(
                    "Manipulate (skipped in replay planner mode): %s", subtask.prompt
                )

        logging.info(
            "Replay planner completed successfully at replay step %d/%d using camera %s.",
            planner.current_step,
            environment.num_steps,
            environment.front_camera_name,
        )
    finally:
        visualizer.close()
        environment.close()


def _run_replay_hybrid(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import (
        replay_manipulation_planner as _replay_manipulation_planner,
    )
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error(
            "--replay-dataset with --replay-mode=hybrid currently supports only --num-episodes=1."
        )
        return

    if not prompt:
        logging.error(
            "--replay-mode=hybrid with --replay-dataset requires a non-empty --prompt."
        )
        return

    if args.replay_manipulate_max_steps <= 0:
        logging.error(
            "--replay-manipulate-max-steps must be positive in hybrid replay mode."
        )
        return

    if args.replay_manipulate_replan_interval_steps <= 0:
        logging.error(
            "--replay-manipulate-replan-interval-steps must be positive in hybrid replay mode."
        )
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay hybrid mode.")

    if args.visualize_playback_rate <= 0:
        logging.error("--visualize-playback-rate must be positive.")
        return

    args.planner.validate_service_config()
    args.planner.validate_motion_limits()
    if not _run_required_server_checks(
        args, needs_pi0=not args.navigation_only, needs_planner=True
    ):
        return

    logging.info("Replay hybrid mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
        max_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    ordered_task_memory_runtime = _build_ordered_task_memory_runtime(
        args,
        environment,
    )
    subtask_list = _build_replay_subtask_list(
        args,
        prompt,
        ordered_task_memory_runtime=ordered_task_memory_runtime,
    )
    if subtask_list is None:
        logging.error(
            "Replay hybrid exiting before execution at replay step %d/%d: subtask decomposition failed.",
            environment.get_cursor(),
            environment.num_steps,
        )
        environment.close()
        return

    visualizer = _replay_visualizer.ReplayVisualizer(
        environment,
        enabled=args.visualize,
        port=args.visualize_port,
        save_path=args.save_path,
    )
    on_nav_step = _build_navigation_visualizer_updater(
        environment,
        visualizer,
        playback_rate=args.visualize_playback_rate,
        enabled=args.visualize,
    )

    policy_agent = None if args.navigation_only else _create_policy_agent(args)
    manipulation_planner = (
        None
        if args.navigation_only
        else (
            _replay_manipulation_planner.ReplayManipulationPromptPlanner(
                environment,
                args.planner,
                task_memory_runtime=ordered_task_memory_runtime,
            )
            if ordered_task_memory_runtime is not None
            else _replay_manipulation_planner.ReplayManipulationPromptPlanner(
                environment,
                args.planner,
            )
        )
    )
    completed_navigate = 0
    completed_manipulate = 0
    policy_steps = 0
    prompt_queries = 0
    debug_export_dir = _resolve_replay_debug_export_dir(args)

    def _log_hybrid_early_exit(
        reason: str, *, subtask_index: int | None = None
    ) -> None:
        if subtask_index is None:
            logging.error(
                "Replay hybrid exiting early at replay step %d/%d: %s",
                environment.get_cursor(),
                environment.num_steps,
                reason,
            )
            return

        logging.error(
            "Replay hybrid exiting early at replay step %d/%d during subtask %d/%d: %s",
            environment.get_cursor(),
            environment.num_steps,
            subtask_index + 1,
            len(subtask_list),
            reason,
        )

    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing replay subtask %d/%d [%s]: %s",
                idx + 1,
                len(subtask_list),
                subtask.type,
                subtask.prompt,
            )
            visualizer.set_subtask_context(
                idx + 1, len(subtask_list), subtask.type, subtask.prompt
            )

            if subtask.type == "navigate":
                if not on_nav_step(_get_visualizer_step(environment)):
                    _log_hybrid_early_exit(
                        "user aborted before navigation execution",
                        subtask_index=idx,
                    )
                    logging.info(
                        "Replay hybrid aborted by user before subtask %d/%d.",
                        idx + 1,
                        len(subtask_list),
                    )
                    return
                completed_navigate += 1
                logging.info(
                    "Skipping navigate subtask %d/%d in replay mode: %s",
                    idx + 1,
                    len(subtask_list),
                    subtask.prompt,
                )
                _mark_ordered_task_completed(
                    ordered_task_memory_runtime,
                    idx,
                    reason=f"replay navigate subtask {idx + 1} skipped/succeeded",
                )
                continue

            if args.navigation_only:
                logging.info("Manipulate (skipped): %s", subtask.prompt)
                continue

            assert (
                policy_agent is not None
            ), "hybrid manipulate subtask requires a policy agent"
            assert (
                manipulation_planner is not None
            ), "hybrid manipulate subtask requires a prompt replanner"
            manipulation_result = _run_manipulation_subtask(
                environment,
                policy_agent,
                manipulation_planner,
                subtask_prompt=subtask.prompt,
                max_steps=args.replay_manipulate_max_steps,
                replan_interval_steps=args.replay_manipulate_replan_interval_steps,
                progress_complete_threshold=args.planner.progress_complete_threshold,
                progress_stall_threshold=args.planner.progress_stall_threshold,
                progress_stall_steps=args.planner.progress_stall_steps,
                progress_regression_threshold=args.planner.progress_regression_threshold,
                progress_confirm_with_replanner=args.planner.progress_confirm_with_replanner,
                progress_head_mode=args.progress_head_mode,
                debug_export_dir=debug_export_dir,
                subtask_index=idx + 1,
                total_subtasks=len(subtask_list),
                visualizer=visualizer,
            )
            executed_steps = int(manipulation_result["executed_steps"])
            policy_steps += executed_steps
            prompt_queries += int(manipulation_result["prompt_queries"])
            if bool(manipulation_result.get("completed", False)):
                completed_manipulate += 1
                logging.info(
                    "Replay manipulate subtask %d/%d completed after %d policy steps, "
                    "%d prompt queries, at replay cursor %d/%d.",
                    idx + 1,
                    len(subtask_list),
                    executed_steps,
                    int(manipulation_result["prompt_queries"]),
                    environment.get_cursor(),
                    environment.num_steps,
                )
                _mark_ordered_task_completed(
                    ordered_task_memory_runtime,
                    idx,
                    reason=f"replay manipulate subtask {idx + 1} completed",
                )
            else:
                stop_reason = str(manipulation_result.get("stop_reason", "incomplete"))
                if stop_reason == "replay_exhausted" and idx + 1 == len(subtask_list):
                    logging.warning(
                        "Replay dataset exhausted during final manipulate subtask %d/%d "
                        "after %d policy steps with no explicit completion signal.",
                        idx + 1,
                        len(subtask_list),
                        executed_steps,
                    )
                    break
                _log_hybrid_early_exit(
                    f"manipulate subtask did not complete ({stop_reason})",
                    subtask_index=idx,
                )
                logging.error(
                    "Replay manipulate subtask %d/%d did not complete after %d policy steps, "
                    "%d prompt queries, at replay cursor %d/%d; stop_reason=%s",
                    idx + 1,
                    len(subtask_list),
                    executed_steps,
                    int(manipulation_result["prompt_queries"]),
                    environment.get_cursor(),
                    environment.num_steps,
                    stop_reason,
                )
                return
            if environment.is_episode_complete() and idx + 1 < len(subtask_list):
                _log_hybrid_early_exit(
                    "replay dataset exhausted with remaining subtasks",
                    subtask_index=idx,
                )
                logging.error(
                    "Replay dataset exhausted after subtask %d/%d; aborting remaining subtasks.",
                    idx + 1,
                    len(subtask_list),
                )
                return

        _log_hybrid_replay_summary(
            environment,
            total_subtasks=len(subtask_list),
            navigate_subtasks=completed_navigate,
            manipulate_subtasks=completed_manipulate,
            policy_steps=policy_steps,
            prompt_queries=prompt_queries,
        )
    finally:
        visualizer.close()
        environment.close()


def _log_real_hybrid_summary(
    *,
    total_subtasks: int,
    navigate_subtasks: int,
    manipulate_subtasks: int,
    policy_steps: int,
    prompt_queries: int,
) -> None:
    logging.info(
        "Real hybrid completed: subtasks=%d, navigate=%d, manipulate=%d, "
        "policy_steps=%d, prompt_queries=%d",
        total_subtasks,
        navigate_subtasks,
        manipulate_subtasks,
        policy_steps,
        prompt_queries,
    )


def _run_real_hybrid(args: Args, prompt: str) -> None:
    """Real-robot two-layer LLM planner loop.

    Mirrors ``_run_replay_hybrid`` but drives ``PiperRealEnvironment`` and calls
    ``navigation_tool.navigate`` for navigate subtasks. Shares the manipulation
    inner loop (``_run_manipulation_subtask``), the VLM prompt replanner
    (``ReplayManipulationPromptPlanner``) and the ordered task-memory runtime
    (``ReplayOrderedTaskMemoryRuntime``).
    """

    import einops
    from openpi_client import image_tools

    from examples.piper_real import base_safety as _base_safety
    from examples.piper_real import env as _env
    from examples.piper_real import logger as _logger
    from examples.piper_real import navigation_tool as _navigation_tool
    from examples.piper_real import (
        replay_manipulation_planner as _replay_manipulation_planner,
    )

    logging.info(
        "Real hybrid entry: pid=%s source=%s manipulate_max_steps=%d "
        "replan_interval_steps=%d max_episode_steps=%d task_spec=%s "
        "robot_base_topic=%s robot_base_cmd_topic=%s",
        os.getpid(),
        Path(__file__).resolve(),
        args.replay_manipulate_max_steps,
        args.replay_manipulate_replan_interval_steps,
        args.max_episode_steps,
        args.planner.task_spec_path or "<unset>",
        args.robot_base_topic,
        args.robot_base_cmd_topic,
    )

    if args.replay_manipulate_max_steps <= 0:
        logging.error(
            "--replay-manipulate-max-steps must be positive in real hybrid mode."
        )
        return

    if args.replay_manipulate_replan_interval_steps <= 0:
        logging.error(
            "--replay-manipulate-replan-interval-steps must be positive in real hybrid mode."
        )
        return

    if args.use_robot_base:
        args.robot_base_topic = args.robot_base_topic.strip()
        args.robot_base_cmd_topic = args.robot_base_cmd_topic.strip()
        if not args.robot_base_topic:
            logging.error("--robot-base-topic must be non-empty when --use-robot-base.")
            return
        if not args.robot_base_cmd_topic:
            logging.error("--robot-base-cmd-topic must be non-empty when --use-robot-base.")
            return

    args.planner.validate_service_config()
    args.planner.validate_motion_limits()

    if not _run_required_server_checks(args, needs_planner=True):
        return

    # Step 1: Decompose task up-front so we know what (if anything) we need to
    # spin up downstream.
    from examples.piper_real import task_decomposer as _task_decomposer

    decomposer = _task_decomposer.TaskDecomposer(args.planner)
    try:
        subtask_list = decomposer.decompose(prompt)
    except _task_decomposer.DecompositionError as exc:
        logging.error("Task decomposition failed: %s", exc)
        return

    has_navigate = any(s.type == "navigate" for s in subtask_list)
    has_manipulate = any(s.type == "manipulate" for s in subtask_list)
    needs_server = has_manipulate and not args.navigation_only
    needs_ros_environment = needs_server or (args.use_robot_base and has_navigate)

    # Step 2: Safety confirmation (once, if base motion requested).
    if args.use_robot_base and has_navigate:
        if not _base_safety.confirm_base_motion_safety(
            prompt,
            use_llm_planner=True,
            # pass False to suppress misleading "policy-driven base control" label
            use_robot_base=False,
        ):
            logging.error("Base motion aborted before execution.")
            return

    # Step 3: Create pi0 connection + policy agent if needed.
    policy_agent = None
    reset_position = None
    if needs_server:
        if not _run_required_server_checks(args, needs_pi0=True):
            return
        policy_agent = _create_policy_agent(args)
        reset_position = getattr(policy_agent, "policy_metadata", {}).get("reset_pose")

    # Step 4: Create shared ROS environment if needed.
    environment = None
    if needs_ros_environment:
        if args.save_log and needs_server:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=reset_position,
            prompt=prompt,
            robot_base_topic=args.robot_base_topic,
            robot_base_cmd_topic=args.robot_base_cmd_topic,
        )
        _restore_cli_logging()
        logging.info("Real hybrid ROS environment initialized.")

    # Step 5: Initialize the real environment before any VLM frame read or
    # policy action. Replay environments can read frame 0 directly; real
    # hardware needs reset() to populate the initial timestep.
    if needs_server and environment is not None:
        environment.reset()
        _restore_cli_logging()
        logging.info("Real hybrid environment reset complete.")
        if policy_agent is not None:
            policy_agent.reset()
            logging.info("Real hybrid policy agent reset complete.")

    # Step 6: Build ordered task memory + manipulation replanner (manipulate only).
    manipulation_planner = None
    ordered_task_memory_runtime = None
    if needs_server and environment is not None:
        ordered_task_memory_runtime = _build_ordered_task_memory_runtime(
            args,
            environment,
        )
        manipulation_planner = (
            _replay_manipulation_planner.ReplayManipulationPromptPlanner(
                environment,
                args.planner,
                task_memory_runtime=ordered_task_memory_runtime,
            )
            if ordered_task_memory_runtime is not None
            else _replay_manipulation_planner.ReplayManipulationPromptPlanner(
                environment,
                args.planner,
            )
        )
        logging.info("Real hybrid manipulation replanner initialized.")

    completed_navigate = 0
    completed_manipulate = 0
    policy_steps = 0
    prompt_queries = 0
    navigation_only_ran = False

    def _dump_nav_frame() -> None:
        if environment is None or not getattr(environment, "save_obs", False):
            return
        ros_op = environment.ros_operator
        if not (
            ros_op.img_front_deque
            and ros_op.img_left_deque
            and ros_op.img_right_deque
        ):
            return
        bridge = ros_op.bridge
        front = bridge.imgmsg_to_cv2(ros_op.img_front_deque[-1], "rgb8")
        left = bridge.imgmsg_to_cv2(ros_op.img_left_deque[-1], "rgb8")
        right = bridge.imgmsg_to_cv2(ros_op.img_right_deque[-1], "rgb8")

        def _to_chw_224(arr):
            resized = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(arr, 224, 224)
            )
            return einops.rearrange(resized, "h w c -> c h w")

        images = {
            "cam_high": _to_chw_224(front),
            "cam_left_wrist": _to_chw_224(left),
            "cam_right_wrist": _to_chw_224(right),
        }
        environment.frame_cnt += 1
        environment.saver.save_images_to_folder(
            images, frame_id=environment.frame_cnt
        )

    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing subtask %d/%d [%s]: %s",
                idx + 1,
                len(subtask_list),
                subtask.type,
                subtask.prompt,
            )

            if subtask.type == "navigate":
                if args.navigation_only and navigation_only_ran:
                    logging.info(
                        "Skipping additional navigate subtask %d/%d in navigation-only mode: %s",
                        idx + 1,
                        len(subtask_list),
                        subtask.prompt,
                    )
                    continue

                ros_operator = None if environment is None else environment.ros_operator
                result = _navigation_tool.navigate(
                    subtask.prompt,
                    ros_operator,
                    dry_run=not args.use_robot_base,
                    frame_tick_callback=_dump_nav_frame,
                )
                if not result.ok:
                    logging.error(
                        "Navigation failed at subtask %d/%d: %s",
                        idx + 1,
                        len(subtask_list),
                        result.error or "unknown error",
                    )
                    return
                navigation_only_ran = True
                completed_navigate += 1
                logging.info(
                    "Navigate subtask %d/%d succeeded via routine %s.",
                    idx + 1,
                    len(subtask_list),
                    result.routine_name,
                )
                _mark_ordered_task_completed(
                    ordered_task_memory_runtime,
                    idx,
                    reason=f"navigate subtask {idx + 1} succeeded",
                )
                continue

            if args.navigation_only:
                logging.info("Manipulate (skipped): %s", subtask.prompt)
                continue

            assert (
                policy_agent is not None and manipulation_planner is not None
                and environment is not None
            ), "manipulate subtask requires pi0 policy agent + manipulation planner + env"

            environment.set_prompt(subtask.prompt)
            _refresh_environment_observation_cache(
                environment,
                context=f"manipulate subtask {idx + 1}/{len(subtask_list)}",
            )
            manipulation_result = _run_manipulation_subtask(
                environment,
                policy_agent,
                manipulation_planner,
                subtask_prompt=subtask.prompt,
                max_steps=args.replay_manipulate_max_steps,
                replan_interval_steps=args.replay_manipulate_replan_interval_steps,
                progress_complete_threshold=args.planner.progress_complete_threshold,
                progress_stall_threshold=args.planner.progress_stall_threshold,
                progress_stall_steps=args.planner.progress_stall_steps,
                progress_regression_threshold=args.planner.progress_regression_threshold,
                progress_confirm_with_replanner=args.planner.progress_confirm_with_replanner,
                progress_head_mode=args.progress_head_mode,
                debug_export_dir="",
                subtask_index=idx + 1,
                total_subtasks=len(subtask_list),
                visualizer=None,
            )
            executed_steps = int(manipulation_result["executed_steps"])
            policy_steps += executed_steps
            prompt_queries += int(manipulation_result["prompt_queries"])
            if bool(manipulation_result.get("completed", False)):
                completed_manipulate += 1
                logging.info(
                    "Manipulate subtask %d/%d completed after %d policy steps, "
                    "%d prompt queries.",
                    idx + 1,
                    len(subtask_list),
                    executed_steps,
                    int(manipulation_result["prompt_queries"]),
                )
                _mark_ordered_task_completed(
                    ordered_task_memory_runtime,
                    idx,
                    reason=f"manipulate subtask {idx + 1} completed",
                )
            else:
                stop_reason = str(manipulation_result.get("stop_reason", "incomplete"))
                logging.error(
                    "Manipulate subtask %d/%d did not complete after %d policy steps, "
                    "%d prompt queries; stop_reason=%s",
                    idx + 1,
                    len(subtask_list),
                    executed_steps,
                    int(manipulation_result["prompt_queries"]),
                    stop_reason,
                )
                return

        _log_real_hybrid_summary(
            total_subtasks=len(subtask_list),
            navigate_subtasks=completed_navigate,
            manipulate_subtasks=completed_manipulate,
            policy_steps=policy_steps,
            prompt_queries=prompt_queries,
        )
    finally:
        save_dir_for_stitch = None
        if environment is not None:
            if args.use_robot_base:
                _base_safety.stop_base(environment.ros_operator)
            if getattr(environment, "saver", None) is not None:
                save_dir_for_stitch = environment.saver.save_dir
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        if save_dir_for_stitch is not None:
            try:
                logging.info(
                    "Stitching camera videos in %s ...", save_dir_for_stitch
                )
                _logger.stitch_camera_videos(save_dir_for_stitch)
            except Exception as exc:  # noqa: BLE001
                logging.warning("stitch_camera_videos failed: %s", exc)


def main(args: Args) -> None:
    logging.info(
        "Piper main source: pid=%s source=%s",
        os.getpid(),
        Path(__file__).resolve(),
    )
    prompt = args.prompt.strip()

    if args.replay_dataset:
        try:
            replay_mode = _resolve_replay_mode(args)
        except ValueError as exc:
            logging.error("%s", exc)
            return

        if args.use_robot_base:
            logging.error(
                "--use-robot-base and --replay-dataset are mutually exclusive."
            )
            return

        if replay_mode == "planner":
            _run_replay_planner(args, prompt)
            return

        if replay_mode == "hybrid":
            _run_replay_hybrid(args, prompt)
            return

        if args.navigation_only:
            logging.error(
                "--navigation-only requires replay planner mode "
                "(--replay-mode planner|hybrid or legacy --use-llm-planner)."
            )
            return

        _run_replay_inference(args, prompt)
        return

    if args.navigation_only and not args.use_llm_planner:
        logging.error("--navigation-only requires --use-llm-planner.")
        return

    if args.use_robot_base and not args.use_llm_planner:
        logging.error("--use-robot-base requires --use-llm-planner.")
        return

    from examples.piper_real import base_safety as _base_safety
    from examples.piper_real import env as _env
    from examples.piper_real import logger as _logger
    from examples.piper_real import navigation_tool as _navigation_tool
    from examples.piper_real import task_decomposer as _task_decomposer
    from openpi_client.runtime import runtime as _runtime

    # ── Stationary manipulation (no LLM planner, or LLM planner with empty prompt) ──
    if not args.use_llm_planner or not prompt:
        if args.use_llm_planner:
            args.planner.validate_service_config()
            logging.info(
                "LLM planner enabled but prompt is empty; running stationary manipulation."
            )

        if not _run_required_server_checks(args, needs_pi0=True):
            return

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

        if args.save_log:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=args.prompt,
            robot_base_topic=args.robot_base_topic,
            robot_base_cmd_topic=args.robot_base_cmd_topic,
        )

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
        return

    # ── Two-layer LLM planner (real robot) ───────────────────────────
    _run_real_hybrid(args, prompt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
