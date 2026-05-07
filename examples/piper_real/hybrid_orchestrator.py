"""Shared two-layer hybrid planning orchestration."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass
class HybridManipulationConfig:
    max_steps: int
    replan_interval_steps: int
    progress_complete_threshold: float
    progress_stall_threshold: float
    progress_stall_steps: int
    progress_regression_threshold: float
    progress_confirm_with_replanner: bool
    progress_head_mode: str
    debug_export_dir: str = ""


@dataclasses.dataclass
class HybridSubtaskResult:
    ok: bool
    stop_reason: str = ""
    episode_complete: bool = False
    executed_steps: int = 0
    prompt_queries: int = 0
    completed: bool = True


@dataclasses.dataclass
class HybridOrchestratorSummary:
    total_subtasks: int = 0
    navigate_subtasks: int = 0
    manipulate_subtasks: int = 0
    policy_steps: int = 0
    prompt_queries: int = 0
    status: str = "completed"
    stop_reason: str = ""


def _environment_cursor(environment: Any) -> int | None:
    getter = getattr(environment, "get_cursor", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:  # noqa: BLE001
        return None


def _environment_complete(environment: Any) -> bool:
    checker = getattr(environment, "is_episode_complete", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:  # noqa: BLE001
        return False


def _visualizer_step(environment: Any) -> int:
    num_steps = int(getattr(environment, "num_steps", 0) or 0)
    if num_steps <= 0:
        return 0
    cursor = _environment_cursor(environment)
    if cursor is None:
        cursor = 0
    return min(max(cursor, 0), num_steps - 1)


@dataclasses.dataclass
class HybridBackendBase:
    environment: Any
    policy_agent: Any | None = None
    manipulation_planner: Any | None = None
    ordered_task_memory_runtime: Any | None = None
    navigation_only: bool = False
    log_prefix: str = "Hybrid"
    summary_logger: Callable[[HybridOrchestratorSummary], None] | None = None
    reason_prefix: str = ""
    ordered_reason_prefix: str = "hybrid"
    navigate_completion_label: str = "succeeded"
    manipulate_log_prefix: str = "Manipulate"
    navigation_only_limit_navigate_once: bool = False
    abort_on_episode_complete_with_remaining: bool = False
    allow_final_replay_exhaustion: bool = False
    manipulation_visualizer: Any | None = None

    def prepare_subtask(self, subtask: Any, index: int, total: int) -> None:
        del subtask, index, total

    def execute_navigate(
        self,
        subtask: Any,
        index: int,
        total: int,
    ) -> HybridSubtaskResult:
        del subtask, index, total
        raise NotImplementedError

    def before_manipulate(self, subtask: Any, index: int, total: int) -> None:
        del subtask, index, total

    def finalize(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    def log_summary(self, summary: HybridOrchestratorSummary) -> None:
        if self.summary_logger is not None:
            self.summary_logger(summary)
            return
        logging.info(
            "%s completed: subtasks=%d, navigate=%d, manipulate=%d, "
            "policy_steps=%d, prompt_queries=%d",
            self.log_prefix,
            summary.total_subtasks,
            summary.navigate_subtasks,
            summary.manipulate_subtasks,
            summary.policy_steps,
            summary.prompt_queries,
        )

    def log_early_exit(
        self,
        reason: str,
        *,
        subtask_index: int | None = None,
        total_subtasks: int | None = None,
    ) -> None:
        cursor = _environment_cursor(self.environment)
        num_steps = getattr(self.environment, "num_steps", None)
        cursor_text = (
            f" at replay step {cursor}/{num_steps}"
            if cursor is not None and num_steps is not None
            else ""
        )
        if subtask_index is None:
            logging.error("%s exiting early%s: %s", self.log_prefix, cursor_text, reason)
            return
        logging.error(
            "%s exiting early%s during subtask %d/%d: %s",
            self.log_prefix,
            cursor_text,
            subtask_index + 1,
            total_subtasks or 0,
            reason,
        )

    def simple_completion_reason(self, subtask: Any, display_index: int) -> str:
        task_kind = "navigate" if subtask.type == "navigate" else "manipulate"
        completion = (
            self.navigate_completion_label
            if subtask.type == "navigate"
            else "completed"
        )
        return f"{self.reason_prefix}{task_kind} subtask {display_index} {completion}"

    def ordered_completion_reason(self, completed_ordered_count: int) -> str:
        return (
            f"{self.ordered_reason_prefix} ordered subtask "
            f"{completed_ordered_count + 1} completed"
        )


@dataclasses.dataclass
class ReplayHybridBackend(HybridBackendBase):
    visualizer: Any | None = None
    on_nav_step: Callable[[int], bool] | None = None

    def __post_init__(self) -> None:
        self.log_prefix = "Replay hybrid"
        self.reason_prefix = "replay "
        self.ordered_reason_prefix = "replay"
        self.navigate_completion_label = "skipped/succeeded"
        self.manipulate_log_prefix = "Replay manipulate"
        self.abort_on_episode_complete_with_remaining = True
        self.allow_final_replay_exhaustion = True
        self.manipulation_visualizer = self.visualizer

    def prepare_subtask(self, subtask: Any, index: int, total: int) -> None:
        if self.visualizer is not None:
            self.visualizer.set_subtask_context(index, total, subtask.type, subtask.prompt)

    def execute_navigate(
        self,
        subtask: Any,
        index: int,
        total: int,
    ) -> HybridSubtaskResult:
        if self.on_nav_step is not None and not self.on_nav_step(
            _visualizer_step(self.environment)
        ):
            self.log_early_exit(
                "user aborted before navigation replay skip",
                subtask_index=index - 1,
                total_subtasks=total,
            )
            logging.info(
                "Replay hybrid aborted by user before subtask %d/%d.",
                index,
                total,
            )
            return HybridSubtaskResult(
                ok=False,
                stop_reason="user_abort",
                completed=False,
            )

        logging.info(
            "Skipping navigate subtask %d/%d in replay mode: %s",
            index,
            total,
            subtask.prompt,
        )
        return HybridSubtaskResult(
            ok=True,
            episode_complete=_environment_complete(self.environment),
            executed_steps=0,
            completed=True,
        )

    def finalize(self) -> None:
        if self.visualizer is not None:
            self.visualizer.close()
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


@dataclasses.dataclass
class RealHybridBackend(HybridBackendBase):
    navigate_func: Callable[..., Any] | None = None
    use_robot_base: bool = False
    frame_tick_callback: Callable[[], None] | None = None
    refresh_observation_cache: Callable[[Any], None] | None = None
    stop_base: Callable[[Any], None] | None = None
    stitch_camera_videos: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.log_prefix = "Real hybrid"
        self.reason_prefix = ""
        self.ordered_reason_prefix = "real"
        self.navigate_completion_label = "succeeded"
        self.manipulate_log_prefix = "Manipulate"
        self.navigation_only_limit_navigate_once = True
        self.abort_on_episode_complete_with_remaining = False
        self.allow_final_replay_exhaustion = False
        self.manipulation_visualizer = None

    def execute_navigate(
        self,
        subtask: Any,
        index: int,
        total: int,
    ) -> HybridSubtaskResult:
        if self.navigate_func is None:
            logging.error("Navigation function is unavailable for real hybrid mode.")
            return HybridSubtaskResult(
                ok=False,
                stop_reason="navigation_unavailable",
                completed=False,
            )
        ros_operator = None if self.environment is None else self.environment.ros_operator
        result = self.navigate_func(
            subtask.prompt,
            ros_operator,
            dry_run=not self.use_robot_base,
            frame_tick_callback=self.frame_tick_callback,
        )
        if not result.ok:
            logging.error(
                "Navigation failed at subtask %d/%d: %s",
                index,
                total,
                result.error or "unknown error",
            )
            return HybridSubtaskResult(
                ok=False,
                stop_reason="navigation_failed",
                completed=False,
            )
        logging.info(
            "Navigate subtask %d/%d succeeded via routine %s.",
            index,
            total,
            result.routine_name,
        )
        return HybridSubtaskResult(ok=True, completed=True)

    def before_manipulate(self, subtask: Any, index: int, total: int) -> None:
        assert self.environment is not None
        self.environment.set_prompt(subtask.prompt)
        if self.refresh_observation_cache is not None:
            self.refresh_observation_cache(
                self.environment,
                context=f"manipulate subtask {index}/{total}",
            )

    def finalize(self) -> None:
        save_dir_for_stitch = None
        if self.environment is not None:
            if self.use_robot_base and self.stop_base is not None:
                self.stop_base(self.environment.ros_operator)
            if getattr(self.environment, "saver", None) is not None:
                save_dir_for_stitch = self.environment.saver.save_dir
            close = getattr(self.environment, "close", None)
            if callable(close):
                close()
        if save_dir_for_stitch is not None and self.stitch_camera_videos is not None:
            try:
                logging.info("Stitching camera videos in %s ...", save_dir_for_stitch)
                self.stitch_camera_videos(save_dir_for_stitch)
            except Exception as exc:  # noqa: BLE001
                logging.warning("stitch_camera_videos failed: %s", exc)


def _default_mark_ordered_task_completed(
    ordered_task_memory_runtime: Any,
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


def _default_get_ordered_completed_count(
    ordered_task_memory_runtime: Any,
    *,
    explicit_completed_count: int,
    log_prefix: str,
) -> int:
    task_spec = ordered_task_memory_runtime.task_spec
    completed_count = max(
        int(explicit_completed_count),
        ordered_task_memory_runtime.memory.highest_completed_count(task_spec),
    )
    try:
        decision = ordered_task_memory_runtime.observe()
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "%s could not refresh ordered task progress; "
            "falling back to explicit completed count: %s",
            log_prefix,
            exc,
        )
        return min(completed_count, task_spec.done_index)

    completed_count = max(
        completed_count,
        len(task_spec.normalize_completed_prefix(decision.completed_subtasks)),
    )
    if decision.current_subtask == task_spec.done_label:
        return task_spec.done_index
    return min(completed_count, task_spec.done_index)


def _result_from_manipulation_dict(
    manipulation_result: dict[str, Any],
    *,
    environment: Any,
    completed: bool,
) -> HybridSubtaskResult:
    return HybridSubtaskResult(
        ok=True,
        stop_reason=str(manipulation_result.get("stop_reason", "")),
        episode_complete=_environment_complete(environment),
        executed_steps=int(manipulation_result.get("executed_steps", 0)),
        prompt_queries=int(manipulation_result.get("prompt_queries", 0)),
        completed=completed,
    )


def run_hybrid_orchestrator(
    *,
    prompt: str,
    backend: HybridBackendBase,
    build_subtask_list: Callable[..., list[Any] | None],
    run_manipulation_subtask: Callable[..., dict[str, Any]],
    manipulation_config: HybridManipulationConfig,
    initial_subtasks: list[Any] | None = None,
    mark_ordered_task_completed: Callable[..., None] = _default_mark_ordered_task_completed,
    get_ordered_completed_count: Callable[..., int] | None = None,
) -> HybridOrchestratorSummary:
    """Run the shared navigate/manipulate hybrid sequence."""

    summary = HybridOrchestratorSummary()
    ordered_task_memory_runtime = backend.ordered_task_memory_runtime
    has_ordered_task_spec = ordered_task_memory_runtime is not None and hasattr(
        ordered_task_memory_runtime,
        "task_spec",
    )
    navigation_only_ran = False

    def _build_subtasks(task_prompt: str) -> list[Any] | None:
        if initial_subtasks is not None and not has_ordered_task_spec and task_prompt == prompt:
            return initial_subtasks
        return build_subtask_list(
            task_prompt,
            ordered_task_memory_runtime=ordered_task_memory_runtime,
        )

    def _completed_count(explicit_completed_count: int) -> int:
        if get_ordered_completed_count is not None:
            return get_ordered_completed_count(
                ordered_task_memory_runtime,
                explicit_completed_count=explicit_completed_count,
            )
        return _default_get_ordered_completed_count(
            ordered_task_memory_runtime,
            explicit_completed_count=explicit_completed_count,
            log_prefix=backend.log_prefix,
        )

    def _execute_subtask(subtask: Any, *, display_index: int, display_total: int):
        nonlocal navigation_only_ran
        logging.info(
            "Executing %s subtask %d/%d [%s]: %s",
            backend.log_prefix.lower(),
            display_index,
            display_total,
            subtask.type,
            subtask.prompt,
        )
        backend.prepare_subtask(subtask, display_index, display_total)

        if subtask.type == "navigate":
            if (
                backend.navigation_only
                and backend.navigation_only_limit_navigate_once
                and navigation_only_ran
            ):
                logging.info(
                    "Skipping additional navigate subtask %d/%d in navigation-only mode: %s",
                    display_index,
                    display_total,
                    subtask.prompt,
                )
                return HybridSubtaskResult(ok=True, completed=False)

            result = backend.execute_navigate(subtask, display_index, display_total)
            if result.ok and result.completed:
                summary.navigate_subtasks += 1
                navigation_only_ran = True
            return result

        if backend.navigation_only:
            logging.info("Manipulate (skipped): %s", subtask.prompt)
            return HybridSubtaskResult(
                ok=True,
                episode_complete=_environment_complete(backend.environment),
                completed=False,
            )

        assert (
            backend.policy_agent is not None
            and backend.manipulation_planner is not None
            and backend.environment is not None
        ), "manipulate subtask requires policy agent + manipulation planner + env"

        backend.before_manipulate(subtask, display_index, display_total)
        manipulation_result = run_manipulation_subtask(
            backend.environment,
            backend.policy_agent,
            backend.manipulation_planner,
            subtask_prompt=subtask.prompt,
            max_steps=manipulation_config.max_steps,
            replan_interval_steps=manipulation_config.replan_interval_steps,
            progress_complete_threshold=manipulation_config.progress_complete_threshold,
            progress_stall_threshold=manipulation_config.progress_stall_threshold,
            progress_stall_steps=manipulation_config.progress_stall_steps,
            progress_regression_threshold=manipulation_config.progress_regression_threshold,
            progress_confirm_with_replanner=(
                manipulation_config.progress_confirm_with_replanner
            ),
            progress_head_mode=manipulation_config.progress_head_mode,
            debug_export_dir=manipulation_config.debug_export_dir,
            subtask_index=display_index,
            total_subtasks=display_total,
            visualizer=backend.manipulation_visualizer,
        )
        executed_steps = int(manipulation_result.get("executed_steps", 0))
        prompt_queries = int(manipulation_result.get("prompt_queries", 0))
        summary.policy_steps += executed_steps
        summary.prompt_queries += prompt_queries

        if bool(manipulation_result.get("completed", False)):
            summary.manipulate_subtasks += 1
            logging.info(
                "%s subtask %d/%d completed after %d policy steps, "
                "%d prompt queries.",
                backend.manipulate_log_prefix,
                display_index,
                display_total,
                executed_steps,
                prompt_queries,
            )
            return _result_from_manipulation_dict(
                manipulation_result,
                environment=backend.environment,
                completed=True,
            )

        stop_reason = str(manipulation_result.get("stop_reason", "incomplete"))
        if stop_reason == "replay_exhausted" and backend.allow_final_replay_exhaustion:
            return _result_from_manipulation_dict(
                manipulation_result,
                environment=backend.environment,
                completed=False,
            )

        backend.log_early_exit(
            f"manipulate subtask did not complete ({stop_reason})",
            subtask_index=display_index - 1,
            total_subtasks=display_total,
        )
        cursor = _environment_cursor(backend.environment)
        num_steps = getattr(backend.environment, "num_steps", None)
        cursor_suffix = (
            f", at replay cursor {cursor}/{num_steps}"
            if cursor is not None and num_steps is not None
            else ""
        )
        logging.error(
            "%s subtask %d/%d did not complete after %d policy steps, "
            "%d prompt queries%s; stop_reason=%s",
            backend.manipulate_log_prefix,
            display_index,
            display_total,
            executed_steps,
            prompt_queries,
            cursor_suffix,
            stop_reason,
        )
        return HybridSubtaskResult(
            ok=False,
            stop_reason=stop_reason,
            episode_complete=_environment_complete(backend.environment),
            executed_steps=executed_steps,
            prompt_queries=prompt_queries,
            completed=False,
        )

    def _handle_result(
        result: HybridSubtaskResult,
        *,
        index: int,
        total: int,
        has_remaining_work: bool,
        ordered: bool,
    ) -> str:
        if not result.ok:
            summary.status = "aborted"
            summary.stop_reason = result.stop_reason
            return "abort"
        if result.stop_reason == "replay_exhausted" and backend.allow_final_replay_exhaustion:
            if has_remaining_work:
                backend.log_early_exit(
                    "replay dataset exhausted with remaining subtasks",
                    subtask_index=index,
                    total_subtasks=total,
                )
                logging.error(
                    "Replay dataset exhausted after subtask %d/%d; aborting remaining subtasks.",
                    index + 1,
                    total,
                )
                summary.status = "aborted"
                summary.stop_reason = "replay_exhausted"
                return "abort"
            logging.warning(
                "Replay dataset exhausted during final manipulate subtask %d/%d "
                "after %d policy steps with no explicit completion signal.",
                index + 1,
                total,
                result.executed_steps,
            )
            return "done" if ordered else "break"
        if (
            result.episode_complete
            and backend.abort_on_episode_complete_with_remaining
            and has_remaining_work
        ):
            backend.log_early_exit(
                "replay dataset exhausted with remaining subtasks",
                subtask_index=index,
                total_subtasks=total,
            )
            logging.error(
                "Replay dataset exhausted after subtask %d/%d; aborting remaining subtasks.",
                index + 1,
                total,
            )
            summary.status = "aborted"
            summary.stop_reason = "replay_exhausted"
            return "abort"
        return "continue"

    try:
        if not has_ordered_task_spec:
            subtask_list = _build_subtasks(prompt)
            if subtask_list is None:
                backend.log_early_exit("subtask decomposition failed")
                summary.status = "aborted"
                summary.stop_reason = "decomposition_failed"
                return summary

            summary.total_subtasks = len(subtask_list)
            for idx, subtask in enumerate(subtask_list):
                result = _execute_subtask(
                    subtask,
                    display_index=idx + 1,
                    display_total=len(subtask_list),
                )
                status = _handle_result(
                    result,
                    index=idx,
                    total=len(subtask_list),
                    has_remaining_work=idx + 1 < len(subtask_list),
                    ordered=False,
                )
                if status == "abort":
                    return summary
                if status == "break":
                    break
                if ordered_task_memory_runtime is not None and result.completed:
                    mark_ordered_task_completed(
                        ordered_task_memory_runtime,
                        idx,
                        reason=backend.simple_completion_reason(subtask, idx + 1),
                    )
        else:
            task_spec = ordered_task_memory_runtime.task_spec
            completed_ordered_count = _completed_count(0)
            summary.total_subtasks = task_spec.done_index
            finished_on_final_replay_exhausted = False

            while completed_ordered_count < task_spec.done_index:
                ordered_prompt = task_spec.next_pending_label(completed_ordered_count)
                logging.info(
                    "%s ordered planning batch %d/%d: %s",
                    backend.log_prefix,
                    completed_ordered_count + 1,
                    task_spec.done_index,
                    ordered_prompt,
                )
                subtask_list = _build_subtasks(ordered_prompt)
                if subtask_list is None:
                    backend.log_early_exit(
                        "subtask decomposition failed",
                        subtask_index=completed_ordered_count,
                        total_subtasks=task_spec.done_index,
                    )
                    summary.status = "aborted"
                    summary.stop_reason = "decomposition_failed"
                    return summary
                if not subtask_list:
                    backend.log_early_exit(
                        "decomposition returned no executable subtasks",
                        subtask_index=completed_ordered_count,
                        total_subtasks=task_spec.done_index,
                    )
                    summary.status = "aborted"
                    summary.stop_reason = "decomposition_empty"
                    return summary

                for batch_idx, subtask in enumerate(subtask_list):
                    result = _execute_subtask(
                        subtask,
                        display_index=completed_ordered_count + 1,
                        display_total=task_spec.done_index,
                    )
                    status = _handle_result(
                        result,
                        index=completed_ordered_count,
                        total=task_spec.done_index,
                        has_remaining_work=(
                            batch_idx + 1 < len(subtask_list)
                            or completed_ordered_count + 1 < task_spec.done_index
                        ),
                        ordered=True,
                    )
                    if status == "abort":
                        return summary
                    if status == "done":
                        finished_on_final_replay_exhausted = True
                        break

                if finished_on_final_replay_exhausted:
                    break
                mark_ordered_task_completed(
                    ordered_task_memory_runtime,
                    completed_ordered_count,
                    reason=backend.ordered_completion_reason(completed_ordered_count),
                )
                completed_ordered_count = _completed_count(
                    completed_ordered_count + 1
                )

        backend.log_summary(summary)
        return summary
    finally:
        backend.finalize()
