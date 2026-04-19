"""TRACER coordinate-based navigation demo."""

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

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
DEFAULT_MAX_LINEAR_VEL_MPS = 0.3
DEFAULT_MAX_ANGULAR_VEL_RAD_S = 0.5


@dataclass(frozen=True)
class NavigationGoal:
    """Represents a target pose in the odom frame."""

    x: float
    y: float
    yaw: float | None = None # yaw angle in radians; if None, only the position is considered for reaching the goal


@dataclass
class NavigationConfig:
    """Stores tunable parameters for coordinate navigation control."""

    control_hz: float = DEFAULT_CONTROL_HZ
    goal_timeout_s: float = DEFAULT_GOAL_TIMEOUT_S
    odom_wait_timeout_s: float = DEFAULT_ODOM_WAIT_TIMEOUT_S 
    # how long to wait for initial odometry before giving up
    goal_hold_seconds: float = DEFAULT_GOAL_HOLD_SECONDS
    position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M 
    # the distance threshold to consider the position goal reached
    yaw_tolerance_rad: float = DEFAULT_YAW_TOLERANCE_RAD
    linear_gain: float = DEFAULT_LINEAR_GAIN
    angular_gain: float = DEFAULT_ANGULAR_GAIN
    heading_align_threshold_rad: float = DEFAULT_HEADING_ALIGN_THRESHOLD_RAD
    max_linear_vel_mps: float = DEFAULT_MAX_LINEAR_VEL_MPS
    max_angular_vel_rad_s: float = DEFAULT_MAX_ANGULAR_VEL_RAD_S

    def validate(self) -> None:
        """Validate navigation parameters and safety limits."""

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
class CoordinateNavigationResult:
    """Captures the outcome and telemetry of a navigation run."""

    ok: bool
    goal: NavigationGoal
    executed_steps: int
    final_pose: dict[str, float] | None = None
    error: str | None = None


@dataclass
class TracerCoordinateRosOperator:
    """Wrap ROS publishers/subscribers needed for base motion control."""

    publisher: object
    twist_type: type
    odom_messages: deque = field(default_factory=lambda: deque(maxlen=1))

    def robot_base_publish(self, values) -> None:
        """Publish a base velocity command as a Twist message."""

        twist = self.twist_type()
        twist.linear.x = float(values[0])
        twist.angular.z = float(values[1])
        self.publisher.publish(twist)

    def robot_base_callback(self, msg) -> None:
        """Cache the latest odometry message from ROS callbacks."""

        self.odom_messages.append(msg)

    def latest_odometry(self) -> dict[str, float] | None:
        """Return the most recent odometry pose as x/y/yaw."""

        if not self.odom_messages:
            return None

        odom = self.odom_messages[-1]
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        return {"x": float(position.x), "y": float(position.y), "yaw": float(yaw)}


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to the range [-pi, pi]."""

    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value to the inclusive [lower, upper] range."""

    return max(lower, min(upper, value))


def _goal_summary(goal: NavigationGoal) -> str:
    """Format a goal for human-readable logging."""

    if goal.yaw is None:
        return f"x={goal.x:.3f}, y={goal.y:.3f}"
    return f"x={goal.x:.3f}, y={goal.y:.3f}, yaw={goal.yaw:.3f}"


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for coordinate navigation arguments."""

    parser = argparse.ArgumentParser(description="Run a TRACER coordinate navigation demo.")
    # add arguments for goal pose, ROS topics, control parameters, and safety limits
    parser.add_argument("--goal-x", type=float, required=True, help="goal x in the odom frame")
    parser.add_argument("--goal-y", type=float, required=True, help="goal y in the odom frame")
    parser.add_argument(
        "--goal-yaw",
        type=float,
        default=None,
        help="optional final yaw in radians; omit to stop after reaching the position",
    )
    parser.add_argument("--odom-topic", default="/odom_raw", help="odometry topic to subscribe to")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="velocity command topic")
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--goal-timeout-s", type=float, default=DEFAULT_GOAL_TIMEOUT_S)
    parser.add_argument("--odom-wait-timeout-s", type=float, default=DEFAULT_ODOM_WAIT_TIMEOUT_S)
    parser.add_argument("--goal-hold-seconds", type=float, default=DEFAULT_GOAL_HOLD_SECONDS)
    parser.add_argument("--position-tolerance-m", type=float, default=DEFAULT_POSITION_TOLERANCE_M)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=DEFAULT_YAW_TOLERANCE_RAD)
    parser.add_argument("--linear-gain", type=float, default=DEFAULT_LINEAR_GAIN)
    parser.add_argument("--angular-gain", type=float, default=DEFAULT_ANGULAR_GAIN)
    parser.add_argument(
        "--heading-align-threshold-rad",
        type=float,
        default=DEFAULT_HEADING_ALIGN_THRESHOLD_RAD,
    )
    parser.add_argument("--max-linear-vel-mps", type=float, default=DEFAULT_MAX_LINEAR_VEL_MPS)
    parser.add_argument("--max-angular-vel-rad-s", type=float, default=DEFAULT_MAX_ANGULAR_VEL_RAD_S)
    return parser


def _build_ros_operator(odom_topic: str, cmd_vel_topic: str) -> TracerCoordinateRosOperator:
    """Initialize ROS interfaces for odometry and velocity commands."""

    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    rospy.init_node("tracer_demo_coordinates_node", anonymous=True)
    publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    operator = TracerCoordinateRosOperator(publisher=publisher, twist_type=Twist)
    rospy.Subscriber(odom_topic, Odometry, operator.robot_base_callback, queue_size=1000, tcp_nodelay=True)
    #subscribe to odometry topic with a large queue and tcp_nodelay for low latency
    rospy.sleep(1.0)
    return operator


def _wait_for_odometry(ros_operator: TracerCoordinateRosOperator, timeout_s: float) -> dict[str, float] | None:
    """Wait for odometry data until timeout and return the latest pose."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        odometry = ros_operator.latest_odometry()
        if odometry is not None:
            return odometry # ready odometry received, return it immediately
        time.sleep(0.05)
    return None


# def _compute_command(
#     current_pose: dict[str, float],
#     goal: NavigationGoal,
#     config: NavigationConfig,
# ) -> tuple[float, float, dict[str, float | str], bool]:
#     """Compute base velocity commands and status for the current pose.
#     This is a core function for tracer control.
#     This function culculates the distance and heading error to the goal, 
#     and decides how to command the robot to move toward the goal while respecting the configured tolerances and gains."""

#     dx = goal.x - current_pose["x"]
#     dy = goal.y - current_pose["y"] # compute the distance and heading error to the goal position
#     distance = math.hypot(dx, dy)
#     heading = math.atan2(dy, dx)
#     heading_error = _normalize_angle(heading - current_pose["yaw"]) 

#     if distance <= config.position_tolerance_m: 
#         # situation A: the robot is close enough to the goal position
#         yaw_error = 0.0
#         if goal.yaw is None: 
#             # if the goal does not specify a yaw, consider the goal reached based on position alone
#             telemetry = {
#                 "distance": distance,
#                 "heading_error": heading_error,
#                 "yaw_error": yaw_error,
#                 "phase": "reached",
#             }
#             return 0.0, 0.0, telemetry, True

#         yaw_error = _normalize_angle(goal.yaw - current_pose["yaw"])
#         if abs(yaw_error) <= config.yaw_tolerance_rad:
#             telemetry = {
#                 "distance": distance,
#                 "heading_error": heading_error,
#                 "yaw_error": yaw_error,
#                 "phase": "reached",
#             }
#             return 0.0, 0.0, telemetry, True

#         angular_z = _clamp(
#             config.angular_gain * yaw_error,
#             -config.max_angular_vel_rad_s,
#             config.max_angular_vel_rad_s,
#         )
#         telemetry = {
#             "distance": distance,
#             "heading_error": heading_error,
#             "yaw_error": yaw_error,
#             "phase": "align",
#         }
#         return 0.0, angular_z, telemetry, False
#     # situation B: the robot is far from the goal or has a large heading error, 
#     # prioritize rotation in place to reduce the heading error before moving forward
#     angular_z = _clamp(
#         config.angular_gain * heading_error,
#         -config.max_angular_vel_rad_s,
#         config.max_angular_vel_rad_s,
#     )
#     if abs(heading_error) > config.heading_align_threshold_rad:
#         linear_x = 0.0
#     else:
#         linear_x = min(config.max_linear_vel_mps, config.linear_gain * distance)

#     telemetry = {
#         "distance": distance,
#         "heading_error": heading_error,
#         "yaw_error": 0.0,
#         "phase": "approach",
#     }
#     return linear_x, angular_z, telemetry, False

def _compute_command(
    current_pose: dict[str, float],
    goal: NavigationGoal,
    config: NavigationConfig,
) -> tuple[float, float, dict[str, float | str], bool]:
    """Compute base velocity commands and status for the current pose."""

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

    # --- 最小改动：支持倒退 ---
    # 若目标在车后方（|heading_error| > pi/2），采用倒车策略：
    # 1) 线速度取负
    # 2) 角度误差按“车尾朝向”计算，避免先大角度掉头
    reverse_mode = abs(heading_error) > (math.pi / 2.0)
    if reverse_mode:
        control_heading_error = _normalize_angle(heading_error - math.copysign(math.pi, heading_error))
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


def navigate_to_goal(
    ros_operator: TracerCoordinateRosOperator | None,
    goal: NavigationGoal,
    config: NavigationConfig,
) -> CoordinateNavigationResult:
    """Drive the base toward a goal pose using feedback control.
    This is another core function.
    Perform input and configuration validity checks.
    Wait for the odometry data to be ready.
    Enter the control loop and repeatedly call _compute_command to generate velocity commands.
    Handle success, timeout, and exceptions, while ensuring the robot is stopped at critical moments."""


    logging.info("Coordinate navigation invoked: goal=(%s)", _goal_summary(goal))

    if ros_operator is None:
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=0,
            error="ros_operator is required",
        )

    config.validate()

    initial_odometry = _wait_for_odometry(ros_operator, config.odom_wait_timeout_s)
    if initial_odometry is None: 
        # overtime while waiting for odometry, stop the base and return failure
        base_safety.stop_base(ros_operator)
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=0,
            error="odometry unavailable",
        )

    logging.info(
        "Initial odometry: x=%.3f y=%.3f yaw=%.3f",
        initial_odometry["x"],
        initial_odometry["y"],
        initial_odometry["yaw"],
    )

    period = 1.0 / config.control_hz
    start_time = time.monotonic()
    next_tick = start_time
    hold_start: float | None = None
    executed_steps = 0

    try:
        while True:
            current_pose = ros_operator.latest_odometry()
            if current_pose is None:
                raise RuntimeError("odometry unavailable")

            linear_x, angular_z, telemetry, reached = _compute_command(current_pose, goal, config)
            # log the current pose, goal, telemetry, and command for debugging and analysis; 
            # this is crucial for understanding the robot's behavior and diagnosing issues during navigation
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
                # Debouncing: only consider the goal reached if we have been within the tolerances for a certain duration, 
                # to avoid stopping prematurely due to transient sensor noise or brief disturbances
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

            if time.monotonic() - start_time >= config.goal_timeout_s: # overtime protection
                raise TimeoutError(
                    f"goal was not reached within {config.goal_timeout_s:.1f} seconds"
                )

            next_tick += period
            sleep_s = next_tick - time.monotonic()  # loop continues
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        base_safety.stop_base(ros_operator) # stop first
        logging.error("Coordinate navigation failed after %d steps: %s", executed_steps, exc)
        return CoordinateNavigationResult(
            ok=False,
            goal=goal,
            executed_steps=executed_steps,
            final_pose=current_pose if "current_pose" in locals() else initial_odometry,
            error=str(exc),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, run navigation, and return a process exit code."""

    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)

    goal = NavigationGoal(x=args.goal_x, y=args.goal_y, yaw=args.goal_yaw)
    config = NavigationConfig(
        control_hz=args.control_hz,
        goal_timeout_s=args.goal_timeout_s,
        odom_wait_timeout_s=args.odom_wait_timeout_s,
        goal_hold_seconds=args.goal_hold_seconds,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        linear_gain=args.linear_gain,
        angular_gain=args.angular_gain,
        heading_align_threshold_rad=args.heading_align_threshold_rad,
        max_linear_vel_mps=args.max_linear_vel_mps,
        max_angular_vel_rad_s=args.max_angular_vel_rad_s,
    )

    try:
        ros_operator = _build_ros_operator(args.odom_topic, args.cmd_vel_topic)
        result = navigate_to_goal(ros_operator, goal, config)
    except KeyboardInterrupt:
        logging.warning("tracer_demo_coordinates interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.exception("tracer_demo_coordinates failed before navigation completed: %s", exc)
        return 1

    if not result.ok:
        logging.error("Coordinate navigation failed: %s", result.error)
        return 1

    logging.info(
        "Coordinate navigation completed: steps=%d final_pose=%s",
        result.executed_steps,
        result.final_pose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())