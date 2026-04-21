"""Coordinate closed-loop navigation tool for TRACER base.

Executes a fixed sequence of body-frame goals relative to the robot's pose at
invocation time, driven by odom feedback. Default routine is byte-for-byte
equivalent to ``scripts/run_tracer_demo_sequence_3term.sh``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from examples.piper_real import base_safety


DEFAULT_CONTROL_HZ = 10.0
DEFAULT_GOAL_TIMEOUT_S = 120.0
DEFAULT_ODOM_WAIT_TIMEOUT_S = 10.0
DEFAULT_GOAL_HOLD_SECONDS = 0.5
DEFAULT_POSITION_TOLERANCE_M = 0.12
DEFAULT_YAW_TOLERANCE_RAD = 0.2
DEFAULT_LINEAR_GAIN = 0.8
DEFAULT_ANGULAR_GAIN = 1.8
DEFAULT_HEADING_ALIGN_THRESHOLD_RAD = 0.35
DEFAULT_MAX_LINEAR_VEL_MPS = 0.15
DEFAULT_MAX_ANGULAR_VEL_RAD_S = 0.25

INTER_STEP_SLEEP_S = 1.0
DEFAULT_ROUTINE_NAME = "default_demo_goal_routine"


@dataclass(frozen=True)
class NavigationGoal:
    """Target pose in an odom-aligned frame."""

    x: float
    y: float
    yaw: float | None = None


@dataclass
class NavigationConfig:
    """Tunable parameters for coordinate navigation control."""

    control_hz: float = DEFAULT_CONTROL_HZ
    goal_timeout_s: float = DEFAULT_GOAL_TIMEOUT_S
    odom_wait_timeout_s: float = DEFAULT_ODOM_WAIT_TIMEOUT_S
    goal_hold_seconds: float = DEFAULT_GOAL_HOLD_SECONDS
    position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M
    yaw_tolerance_rad: float = DEFAULT_YAW_TOLERANCE_RAD
    linear_gain: float = DEFAULT_LINEAR_GAIN
    angular_gain: float = DEFAULT_ANGULAR_GAIN
    heading_align_threshold_rad: float = DEFAULT_HEADING_ALIGN_THRESHOLD_RAD
    max_linear_vel_mps: float = DEFAULT_MAX_LINEAR_VEL_MPS
    max_angular_vel_rad_s: float = DEFAULT_MAX_ANGULAR_VEL_RAD_S

    def validate(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.goal_timeout_s <= 0:
            raise ValueError("goal_timeout_s must be positive")
        if self.odom_wait_timeout_s <= 0:
            raise ValueError("odom_wait_timeout_s must be positive")
        if self.goal_hold_seconds < 0:
            raise ValueError("goal_hold_seconds must be non-negative")
        if self.position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be positive")
        if self.yaw_tolerance_rad <= 0:
            raise ValueError("yaw_tolerance_rad must be positive")
        if self.linear_gain <= 0:
            raise ValueError("linear_gain must be positive")
        if self.angular_gain <= 0:
            raise ValueError("angular_gain must be positive")
        if self.heading_align_threshold_rad <= 0:
            raise ValueError("heading_align_threshold_rad must be positive")
        if self.max_linear_vel_mps <= 0:
            raise ValueError("max_linear_vel_mps must be positive")
        if self.max_linear_vel_mps > base_safety.TRACER_MANUAL_MAX_LINEAR_VEL_MPS:
            raise ValueError(
                f"max_linear_vel_mps must be <= {base_safety.TRACER_MANUAL_MAX_LINEAR_VEL_MPS} m/s"
            )
        if self.max_angular_vel_rad_s <= 0:
            raise ValueError("max_angular_vel_rad_s must be positive")
        if self.max_angular_vel_rad_s > base_safety.TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S:
            raise ValueError(
                f"max_angular_vel_rad_s must be <= {base_safety.TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S} rad/s"
            )


@dataclass
class NavigationResult:
    """Aggregate outcome of a full routine run."""

    ok: bool
    prompt: str
    routine_name: str
    executed_steps: int
    error: str | None = None


@dataclass
class CoordinateNavigationResult:
    """Per-goal outcome produced by :func:`navigate_to_goal`."""

    ok: bool
    goal: NavigationGoal
    executed_steps: int
    final_pose: dict[str, float] | None = None
    error: str | None = None


@dataclass
class TracerCoordinateRosOperator:
    """ROS publisher/subscriber adapter matching the coordinate control loop."""

    publisher: object
    twist_type: type
    odom_messages: deque = field(default_factory=lambda: deque(maxlen=1))

    def robot_base_publish(self, values) -> None:
        twist = self.twist_type()
        twist.linear.x = float(values[0])
        twist.angular.z = float(values[1])
        self.publisher.publish(twist)

    def robot_base_callback(self, msg) -> None:
        self.odom_messages.append(msg)

    def latest_odometry(self) -> dict[str, float] | None:
        if not self.odom_messages:
            return None

        odom = self.odom_messages[-1]
        return _odom_msg_to_pose(odom)


# 3term-equivalent fixed sequence: body-frame offsets from pose at navigate() start.
DEFAULT_DEMO_GOAL_ROUTINE: tuple[NavigationGoal, ...] = (
    NavigationGoal(x=-0.3, y=0.0, yaw=0.0),
    NavigationGoal(x=-0.3, y=0.0, yaw=math.pi / 2.0),
    NavigationGoal(x=-0.3, y=0.6, yaw=math.pi / 2.0),
    NavigationGoal(x=-0.3, y=0.6, yaw=0.0),
    NavigationGoal(x=0.0, y=0.6, yaw=0.0),
)


def _odom_msg_to_pose(odom) -> dict[str, float]:
    position = odom.pose.pose.position
    orientation = odom.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    return {"x": float(position.x), "y": float(position.y), "yaw": float(yaw)}


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _goal_summary(goal: NavigationGoal) -> str:
    if goal.yaw is None:
        return f"x={goal.x:.3f}, y={goal.y:.3f}"
    return f"x={goal.x:.3f}, y={goal.y:.3f}, yaw={goal.yaw:.3f}"


class _RosOperatorCoordinateAdapter:
    """Adapt a stock ``RosOperator`` (with ``robot_base_deque``) to the coordinate interface."""

    def __init__(self, ros_operator: Any) -> None:
        self._ros_operator = ros_operator

    def robot_base_publish(self, values) -> None:
        self._ros_operator.robot_base_publish(values)

    def latest_odometry(self) -> dict[str, float] | None:
        deque_ = getattr(self._ros_operator, "robot_base_deque", None)
        if not deque_:
            return None
        return _odom_msg_to_pose(deque_[-1])


def _ensure_coordinate_interface(ros_operator: Any) -> Any:
    """Return an object exposing ``robot_base_publish`` and ``latest_odometry``."""

    if hasattr(ros_operator, "latest_odometry"):
        return ros_operator
    return _RosOperatorCoordinateAdapter(ros_operator)


def _wait_for_odometry(ros_operator: Any, timeout_s: float) -> dict[str, float] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pose = ros_operator.latest_odometry()
        if pose is not None:
            return pose
        time.sleep(0.05)
    return None


def _compute_command(
    current_pose: dict[str, float],
    goal: NavigationGoal,
    config: NavigationConfig,
) -> tuple[float, float, dict[str, float | str], bool]:
    """Compute base velocity commands and status for the current pose.

    Supports reverse driving when the goal lies behind the robot
    (|heading_error| > pi/2): uses negative linear speed and aligns the rear
    heading to avoid a large in-place turn.
    """

    dx = goal.x - current_pose["x"]
    dy = goal.y - current_pose["y"]
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    heading_error = _normalize_angle(heading - current_pose["yaw"])

    if distance <= config.position_tolerance_m:
        yaw_error = 0.0
        if goal.yaw is None:
            telemetry = {
                "distance": distance,
                "heading_error": heading_error,
                "yaw_error": yaw_error,
                "phase": "reached",
            }
            return 0.0, 0.0, telemetry, True

        yaw_error = _normalize_angle(goal.yaw - current_pose["yaw"])
        if abs(yaw_error) <= config.yaw_tolerance_rad:
            telemetry = {
                "distance": distance,
                "heading_error": heading_error,
                "yaw_error": yaw_error,
                "phase": "reached",
            }
            return 0.0, 0.0, telemetry, True

        angular_z = _clamp(
            config.angular_gain * yaw_error,
            -config.max_angular_vel_rad_s,
            config.max_angular_vel_rad_s,
        )
        telemetry = {
            "distance": distance,
            "heading_error": heading_error,
            "yaw_error": yaw_error,
            "phase": "align",
        }
        return 0.0, angular_z, telemetry, False

    reverse_mode = abs(heading_error) > (math.pi / 2.0)
    if reverse_mode:
        control_heading_error = _normalize_angle(
            heading_error - math.copysign(math.pi, heading_error)
        )
    else:
        control_heading_error = heading_error

    angular_z = _clamp(
        config.angular_gain * control_heading_error,
        -config.max_angular_vel_rad_s,
        config.max_angular_vel_rad_s,
    )

    if abs(control_heading_error) > config.heading_align_threshold_rad:
        linear_x = 0.0
    else:
        speed = min(config.max_linear_vel_mps, config.linear_gain * distance)
        linear_x = -speed if reverse_mode else speed

    telemetry = {
        "distance": distance,
        "heading_error": heading_error,
        "yaw_error": 0.0,
        "phase": "approach_reverse" if reverse_mode else "approach",
    }
    return linear_x, angular_z, telemetry, False


def _resolve_body_frame_offset(
    origin: dict[str, float], offset: NavigationGoal
) -> NavigationGoal:
    cos_y = math.cos(origin["yaw"])
    sin_y = math.sin(origin["yaw"])
    abs_x = origin["x"] + offset.x * cos_y - offset.y * sin_y
    abs_y = origin["y"] + offset.x * sin_y + offset.y * cos_y
    abs_yaw = (
        _normalize_angle(origin["yaw"] + offset.yaw) if offset.yaw is not None else None
    )
    return NavigationGoal(x=abs_x, y=abs_y, yaw=abs_yaw)


def _resolve_relative_goal(
    ros_operator: Any,
    goal: NavigationGoal,
    odom_wait_timeout_s: float,
) -> NavigationGoal:
    origin = _wait_for_odometry(ros_operator, odom_wait_timeout_s)
    if origin is None:
        raise RuntimeError("odometry unavailable while resolving relative goal")

    resolved = _resolve_body_frame_offset(origin, goal)
    logging.info(
        "Relative goal (dx=%.3f, dy=%.3f, dyaw=%s) from pose (x=%.3f, y=%.3f, yaw=%.3f) "
        "resolved to absolute (x=%.3f, y=%.3f, yaw=%s)",
        goal.x,
        goal.y,
        f"{goal.yaw:.3f}" if goal.yaw is not None else "None",
        origin["x"],
        origin["y"],
        origin["yaw"],
        resolved.x,
        resolved.y,
        f"{resolved.yaw:.3f}" if resolved.yaw is not None else "None",
    )
    return resolved


def navigate_to_goal(
    ros_operator: Any | None,
    goal: NavigationGoal,
    config: NavigationConfig,
) -> CoordinateNavigationResult:
    """Drive the base toward a single goal pose using feedback control."""

    logging.info("Coordinate navigation invoked: goal=(%s)", _goal_summary(goal))

    if ros_operator is None:
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=0,
            error="ros_operator is required",
        )

    config.validate()

    initial_pose = _wait_for_odometry(ros_operator, config.odom_wait_timeout_s)
    if initial_pose is None:
        base_safety.stop_base(ros_operator)
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=0,
            error="odometry unavailable",
        )

    logging.info(
        "Initial odometry: x=%.3f y=%.3f yaw=%.3f",
        initial_pose["x"],
        initial_pose["y"],
        initial_pose["yaw"],
    )

    period = 1.0 / config.control_hz
    start_time = time.monotonic()
    next_tick = start_time
    hold_start: float | None = None
    executed_steps = 0
    current_pose: dict[str, float] | None = initial_pose

    try:
        while True:
            current_pose = ros_operator.latest_odometry()
            if current_pose is None:
                raise RuntimeError("odometry unavailable")

            linear_x, angular_z, telemetry, reached = _compute_command(
                current_pose, goal, config
            )
            logging.info(
                (
                    "goal=(%s) pose=(x=%.3f, y=%.3f, yaw=%.3f) "
                    "distance=%.3f heading_error=%.3f yaw_error=%.3f phase=%s cmd=(%.3f, %.3f)"
                ),
                _goal_summary(goal),
                current_pose["x"],
                current_pose["y"],
                current_pose["yaw"],
                float(telemetry["distance"]),
                float(telemetry["heading_error"]),
                float(telemetry["yaw_error"]),
                telemetry["phase"],
                linear_x,
                angular_z,
            )

            if reached:
                if hold_start is None:
                    hold_start = time.monotonic()
                ros_operator.robot_base_publish([0.0, 0.0])
                executed_steps += 1
                if time.monotonic() - hold_start >= config.goal_hold_seconds:
                    base_safety.stop_base(ros_operator)
                    return CoordinateNavigationResult(
                        ok=True,
                        goal=goal,
                        executed_steps=executed_steps,
                        final_pose=current_pose,
                    )
            else:
                hold_start = None
                linear_x, angular_z = base_safety.enforce_base_velocity_limits(
                    linear_x,
                    angular_z,
                    max_linear_vel=config.max_linear_vel_mps,
                    max_angular_vel=config.max_angular_vel_rad_s,
                    source="coordinate navigation command",
                )
                ros_operator.robot_base_publish([linear_x, angular_z])
                executed_steps += 1

            if time.monotonic() - start_time >= config.goal_timeout_s:
                raise TimeoutError(
                    f"goal was not reached within {config.goal_timeout_s:.1f} seconds"
                )

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        base_safety.stop_base(ros_operator)
        logging.error(
            "Coordinate navigation failed after %d steps: %s", executed_steps, exc
        )
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=executed_steps,
            final_pose=current_pose,
            error=str(exc),
        )


def navigate(
    prompt: str,
    ros_operator: Any | None,
    *,
    dry_run: bool = False,
    routine: tuple[NavigationGoal, ...] = DEFAULT_DEMO_GOAL_ROUTINE,
    routine_name: str = DEFAULT_ROUTINE_NAME,
    config: NavigationConfig | None = None,
    inter_step_sleep_s: float = INTER_STEP_SLEEP_S,
) -> NavigationResult:
    """Run the fixed goal routine against ``ros_operator``.

    The routine entries are interpreted as body-frame offsets from the robot's
    pose at the moment of invocation, matching the accumulation semantics of
    ``scripts/run_tracer_demo_sequence_3term.sh``.
    """

    logging.info(
        "Navigation tool invoked: prompt=%s routine=%s dry_run=%s",
        prompt,
        routine_name,
        dry_run,
    )

    if dry_run or ros_operator is None:
        for idx, goal in enumerate(routine, start=1):
            logging.info(
                "Navigation goal %d/%d (dry_run): %s",
                idx,
                len(routine),
                _goal_summary(goal),
            )
        return NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=0,
        )

    adapted = _ensure_coordinate_interface(ros_operator)
    cfg = config or NavigationConfig()

    try:
        cfg.validate()
    except ValueError as exc:
        base_safety.stop_base(ros_operator)
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=0,
            error=str(exc),
        )

    origin = _wait_for_odometry(adapted, cfg.odom_wait_timeout_s)
    if origin is None:
        base_safety.stop_base(ros_operator)
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=0,
            error="odometry unavailable",
        )

    logging.info(
        "Navigation origin pose: x=%.3f y=%.3f yaw=%.3f",
        origin["x"],
        origin["y"],
        origin["yaw"],
    )

    executed_steps = 0
    try:
        for idx, offset in enumerate(routine, start=1):
            absolute_goal = _resolve_body_frame_offset(origin, offset)
            logging.info(
                "Navigation step %d/%d: offset=(%s) -> absolute=(%s)",
                idx,
                len(routine),
                _goal_summary(offset),
                _goal_summary(absolute_goal),
            )
            result = navigate_to_goal(adapted, absolute_goal, cfg)
            if not result.ok:
                base_safety.stop_base(ros_operator)
                return NavigationResult(
                    ok=False,
                    prompt=prompt,
                    routine_name=routine_name,
                    executed_steps=executed_steps,
                    error=result.error or "goal failed",
                )
            executed_steps = idx
            if inter_step_sleep_s > 0 and idx < len(routine):
                time.sleep(inter_step_sleep_s)
    except Exception as exc:  # noqa: BLE001
        base_safety.stop_base(ros_operator)
        logging.error(
            "Navigation tool failed after %d steps: %s", executed_steps, exc
        )
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=executed_steps,
            error=str(exc),
        )

    base_safety.stop_base(ros_operator)
    return NavigationResult(
        ok=True,
        prompt=prompt,
        routine_name=routine_name,
        executed_steps=executed_steps,
    )
