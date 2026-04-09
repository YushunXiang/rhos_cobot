import argparse
import logging
from dataclasses import dataclass
from typing import Sequence

from examples.piper_real import navigation_tool


@dataclass
class TracerRosOperator:
    publisher: object
    twist_type: type

    def robot_base_publish(self, values) -> None:
        twist = self.twist_type()
        twist.linear.x = float(values[0])
        twist.angular.z = float(values[1])
        self.publisher.publish(twist)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the default TRACER navigation demo routine.")
    parser.add_argument("--prompt", default="demo navigate task")
    return parser


def _build_ros_operator() -> TracerRosOperator:
    import rospy
    from geometry_msgs.msg import Twist

    rospy.init_node("tracer_demo_node", anonymous=True)
    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.sleep(1.0)
    return TracerRosOperator(publisher=publisher, twist_type=Twist)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)

    try:
        ros_operator = _build_ros_operator()
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


