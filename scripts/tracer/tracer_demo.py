import argparse
import logging
from typing import Sequence

from examples.piper_real import navigation_tool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the default TRACER navigation demo routine.")
    parser.add_argument("--prompt", default="demo navigate task")
    parser.add_argument("--odom-topic", default="/odom_raw", help="odometry topic to subscribe to")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="velocity command topic")
    return parser


def _build_ros_operator(
    odom_topic: str, cmd_vel_topic: str
) -> navigation_tool.TracerCoordinateRosOperator:
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    rospy.init_node("tracer_demo_node", anonymous=True)
    publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    operator = navigation_tool.TracerCoordinateRosOperator(publisher=publisher, twist_type=Twist)
    rospy.Subscriber(
        odom_topic,
        Odometry,
        operator.robot_base_callback,
        queue_size=1000,
        tcp_nodelay=True,
    )
    rospy.sleep(1.0)
    return operator


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)

    try:
        ros_operator = _build_ros_operator(args.odom_topic, args.cmd_vel_topic)
        result = navigation_tool.navigate(args.prompt, ros_operator, dry_run=False)
    except Exception as exc:  # noqa: BLE001
        logging.exception("tracer_demo failed before navigation completed: %s", exc)
        return 1

    if not result.ok:
        logging.error("Navigation failed: %s", result.error)
        return 1

    logging.info(
        "Navigation completed: routine=%s executed_steps=%d",
        result.routine_name,
        result.executed_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
