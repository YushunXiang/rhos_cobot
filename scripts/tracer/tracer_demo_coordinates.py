"""TRACER coordinate-based navigation demo (thin CLI wrapper).

The control primitives now live in :mod:`examples.piper_real.navigation_tool`.
This module keeps the original CLI interface used by
``scripts/run_tracer_demo_sequence_3term.sh``.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from examples.piper_real import base_safety  # re-export for tests patching stop_base
from examples.piper_real.navigation_tool import (  # re-export for tests / external callers
    DEFAULT_ANGULAR_GAIN,
    DEFAULT_CONTROL_HZ,
    DEFAULT_GOAL_HOLD_SECONDS,
    DEFAULT_GOAL_TIMEOUT_S,
    DEFAULT_HEADING_ALIGN_THRESHOLD_RAD,
    DEFAULT_LINEAR_GAIN,
    DEFAULT_MAX_ANGULAR_VEL_RAD_S,
    DEFAULT_MAX_LINEAR_VEL_MPS,
    DEFAULT_ODOM_WAIT_TIMEOUT_S,
    DEFAULT_POSITION_TOLERANCE_M,
    DEFAULT_YAW_TOLERANCE_RAD,
    CoordinateNavigationResult,
    NavigationConfig,
    NavigationGoal,
    TracerCoordinateRosOperator,
    _clamp,
    _compute_command,
    _goal_summary,
    _normalize_angle,
    _resolve_relative_goal,
    _wait_for_odometry,
    navigate_to_goal,
)


__all__ = [
    "CoordinateNavigationResult",
    "DEFAULT_ANGULAR_GAIN",
    "DEFAULT_CONTROL_HZ",
    "DEFAULT_GOAL_HOLD_SECONDS",
    "DEFAULT_GOAL_TIMEOUT_S",
    "DEFAULT_HEADING_ALIGN_THRESHOLD_RAD",
    "DEFAULT_LINEAR_GAIN",
    "DEFAULT_MAX_ANGULAR_VEL_RAD_S",
    "DEFAULT_MAX_LINEAR_VEL_MPS",
    "DEFAULT_ODOM_WAIT_TIMEOUT_S",
    "DEFAULT_POSITION_TOLERANCE_M",
    "DEFAULT_YAW_TOLERANCE_RAD",
    "base_safety",
    "NavigationConfig",
    "NavigationGoal",
    "TracerCoordinateRosOperator",
    "_clamp",
    "_compute_command",
    "_goal_summary",
    "_normalize_angle",
    "_resolve_relative_goal",
    "_wait_for_odometry",
    "navigate_to_goal",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TRACER coordinate navigation demo.")
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
    parser.add_argument(
        "--relative",
        action="store_true",
        help="interpret goal as body-frame offset from the robot pose at script start",
    )
    return parser


def _build_ros_operator(odom_topic: str, cmd_vel_topic: str) -> TracerCoordinateRosOperator:
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    rospy.init_node("tracer_demo_coordinates_node", anonymous=True)
    publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    operator = TracerCoordinateRosOperator(publisher=publisher, twist_type=Twist)
    rospy.Subscriber(
        odom_topic, Odometry, operator.robot_base_callback, queue_size=1000, tcp_nodelay=True
    )
    rospy.sleep(1.0)
    return operator


def main(argv: Sequence[str] | None = None) -> int:
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
        if args.relative:
            goal = _resolve_relative_goal(ros_operator, goal, config.odom_wait_timeout_s)
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