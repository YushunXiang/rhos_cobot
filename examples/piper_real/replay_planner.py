"""Offline HDF5 replay adapter for the VLM navigation planner."""

from __future__ import annotations

import collections
import math
from types import SimpleNamespace

from examples.piper_real.llm_planner import LLMNavigationPlanner
from examples.piper_real.llm_planner import PlannerResponseError
from examples.piper_real.planner_config import PlannerConfig
from examples.piper_real.replay_env import ReplayEnvironment


class _PassthroughBridge:
    def imgmsg_to_cv2(self, image, _encoding: str):
        return image


class _ReplayRosOperator:
    def __init__(self) -> None:
        self.bridge = _PassthroughBridge()
        self.img_front_deque = collections.deque(maxlen=1)
        self.robot_base_deque = collections.deque(maxlen=1)
        self.published_commands: list[list[float]] = []

    def robot_base_publish(self, command: list[float]) -> None:
        self.published_commands.append([float(command[0]), float(command[1])])


def _make_fake_odometry(odom: dict[str, float]):
    yaw = float(odom["yaw"])
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    position = SimpleNamespace(
        x=float(odom["x"]),
        y=float(odom["y"]),
        z=0.0,
    )
    pose = SimpleNamespace(position=position, orientation=orientation)
    return SimpleNamespace(pose=SimpleNamespace(pose=pose))


class OfflineReplayNavigationPlanner(LLMNavigationPlanner):
    """Drive the navigation planner with recorded frames instead of live ROS topics."""

    def __init__(self, replay_environment: ReplayEnvironment, config: PlannerConfig) -> None:
        self.replay_environment = replay_environment
        self._replay_step = 0
        self._replay_exhausted = replay_environment.num_steps == 0
        ros_operator = _ReplayRosOperator()
        super().__init__(ros_operator=ros_operator, config=config)
        if not self._replay_exhausted:
            self._load_replay_step(self._replay_step)

    @property
    def current_step(self) -> int:
        return self._replay_step

    @property
    def replay_commands(self) -> list[list[float]]:
        return list(self.ros_operator.published_commands)

    def run(self, task_prompt: str) -> bool:
        if self._replay_exhausted:
            self._log_status(
                "navigation_failed",
                {
                    "reason": "replay dataset exhausted before navigation started",
                    "usable_steps": 0,
                    "replay_step": self._replay_step,
                },
            )
            return False

        if not self.ros_operator.img_front_deque or not self.ros_operator.robot_base_deque:
            self._load_replay_step(self._replay_step)

        return super().run(task_prompt)

    def capture_front_image(self) -> str:
        if self._replay_exhausted:
            raise PlannerResponseError("replay dataset exhausted before planner returned stop")
        return super().capture_front_image()

    def get_odometry(self) -> dict[str, float]:
        if self._replay_exhausted:
            raise PlannerResponseError("replay dataset exhausted before planner returned stop")
        return super().get_odometry()

    def execute_command(self, cmd: dict[str, object]) -> dict[str, object]:
        linear_x = float(cmd["linear_x"])
        angular_z = float(cmd["angular_z"])
        duration = float(cmd.get("duration", self.config.default_duration))
        replay_step_before = self._replay_step
        recorded_base_action = self.replay_environment.get_recorded_base_action(replay_step_before)

        self.ros_operator.robot_base_publish([linear_x, angular_z])
        replay_frames_advanced = self._advance_replay_cursor(duration)
        self.stop_base()

        return {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration,
            "replay_step_before": replay_step_before,
            "replay_step_after": self._replay_step,
            "replay_frames_advanced": replay_frames_advanced,
            "replay_front_camera": self.replay_environment.front_camera_name,
            "recorded_base_action": (
                recorded_base_action.tolist() if recorded_base_action is not None else None
            ),
        }

    def _advance_replay_cursor(self, duration: float) -> int:
        last_step = self.replay_environment.num_steps - 1
        if self._replay_step >= last_step:
            self._mark_replay_exhausted()
            return 0

        fps = self.replay_environment.fps
        replay_frames = max(1, int(round(duration * fps))) if fps > 0 else 1
        next_step = min(self._replay_step + replay_frames, last_step)
        advanced = next_step - self._replay_step
        self._load_replay_step(next_step)
        return advanced

    def _load_replay_step(self, step_idx: int) -> None:
        self._replay_step = step_idx
        self._replay_exhausted = False
        self.ros_operator.img_front_deque.clear()
        self.ros_operator.robot_base_deque.clear()
        self.ros_operator.img_front_deque.append(self.replay_environment.get_front_image(step_idx))
        self.ros_operator.robot_base_deque.append(
            _make_fake_odometry(self.replay_environment.get_estimated_odometry(step_idx))
        )

    def _mark_replay_exhausted(self) -> None:
        self._replay_exhausted = True
        self.ros_operator.img_front_deque.clear()
        self.ros_operator.robot_base_deque.clear()
