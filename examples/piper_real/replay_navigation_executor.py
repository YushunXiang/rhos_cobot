"""Replay navigation executor for hybrid mode.

This runs the shared fixed navigation routine against a replay dataset instead
of asking a VLM for per-step base commands.
"""

from __future__ import annotations

import logging
from typing import Callable

from examples.piper_real import navigation_tool
from examples.piper_real.replay_env import ReplayEnvironment


class ReplayNavigationExecutor:
    def __init__(
        self,
        replay_environment: ReplayEnvironment,
        *,
        on_step_callback: Callable[[int], bool] | None = None,
    ) -> None:
        self.replay_environment = replay_environment
        self._on_step_callback = on_step_callback
        self.published_commands: list[list[float]] = []

    @property
    def current_step(self) -> int:
        return self.replay_environment.get_cursor()

    def navigate(
        self,
        prompt: str,
        *,
        dry_run: bool = False,
    ) -> navigation_tool.NavigationResult:
        return navigation_tool._run_navigation_routine(
            prompt,
            routine_name=navigation_tool.DEFAULT_ROUTINE_NAME,
            routine=navigation_tool.DEFAULT_DEMO_ROUTINE,
            execute_step=self._execute_step,
            stop_base_fn=self.stop_base,
            dry_run=dry_run,
            inter_step_sleep_s=0.0,
        )

    def stop_base(self) -> None:
        self.published_commands.append([0.0, 0.0])

    def _execute_step(self, step: tuple[float, float, float]) -> None:
        if self.replay_environment.get_cursor() >= self.replay_environment.num_steps:
            raise RuntimeError("replay dataset exhausted before navigation step")

        linear_x, angular_z, duration = step
        self.published_commands.append([float(linear_x), float(angular_z)])
        replay_frames_advanced = self._advance_replay_cursor(float(duration))
        self.stop_base()

        if replay_frames_advanced <= 0:
            raise RuntimeError("replay dataset exhausted during navigation step")

        logging.info(
            "Replay navigation step advanced %d frames to replay step %d/%d.",
            replay_frames_advanced,
            self.replay_environment.get_cursor(),
            self.replay_environment.num_steps,
        )

    def _advance_replay_cursor(self, duration: float) -> int:
        current_step = self.replay_environment.get_cursor()
        last_step = self.replay_environment.num_steps - 1
        if current_step >= last_step:
            self.replay_environment.set_cursor(self.replay_environment.num_steps)
            return 0

        fps = self.replay_environment.fps
        replay_frames = max(1, int(round(duration * fps))) if fps > 0 else 1
        next_step = min(current_step + replay_frames, last_step)
        self.replay_environment.set_cursor(next_step)
        self._emit_step_updates(current_step, next_step)
        return next_step - current_step

    def _emit_step_updates(self, previous_step: int, next_step: int) -> None:
        if self._on_step_callback is None:
            return

        if next_step <= previous_step:
            self._on_step_callback(next_step)
            return

        for frame_idx in range(previous_step + 1, next_step + 1):
            if not self._on_step_callback(frame_idx):
                break
