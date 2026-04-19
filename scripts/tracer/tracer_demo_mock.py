import argparse
import logging
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TwistLinear:
    x: float = 0.0


@dataclass
class TwistAngular:
    z: float = 0.0


class Twist:
    def __init__(self):
        self.linear = TwistLinear()
        self.angular = TwistAngular()


class MockPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, twist: Twist):
        self.messages.append((twist.linear.x, twist.angular.z))
        print(f"[MOCK PUBLISH] /cmd_vel linear.x={twist.linear.x:.3f}, angular.z={twist.angular.z:.3f}")


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
    parser = argparse.ArgumentParser(description="TRACER fixed-path mock demo (no LLM, no real robot).")
    parser.add_argument("--hz", type=float, default=10.0, help="publish rate")
    parser.add_argument("--dry-run", action="store_true", help="only print command steps")
    return parser


def build_fixed_routine():
    # (vx, wz, duration_sec)
    # 例子：前进2秒 -> 原地左转2秒 -> 前进2秒 -> 停止
    return [
        (0.30, 0.00, 2.0),
        (0.00, 0.60, 2.0),
        (0.30, 0.00, 2.0),
        (0.00, 0.00, 0.5),
    ]


def execute_routine(operator: TracerRosOperator, routine, hz: float, dry_run: bool):
    dt = 1.0 / hz
    total_steps = 0

    for seg_id, (vx, wz, duration) in enumerate(routine):
        steps = max(1, int(duration * hz))
        logging.info("segment=%d vx=%.3f wz=%.3f duration=%.2fs steps=%d", seg_id, vx, wz, duration, steps)

        for _ in range(steps):
            if dry_run:
                print(f"[DRY RUN] cmd_vel: vx={vx:.3f}, wz={wz:.3f}")
            else:
                operator.robot_base_publish([vx, wz])
            total_steps += 1
            time.sleep(dt)

    return total_steps


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)

    publisher = MockPublisher()
    operator = TracerRosOperator(publisher=publisher, twist_type=Twist)
    routine = build_fixed_routine()

    try:
        steps = execute_routine(operator, routine, hz=args.hz, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logging.exception("mock routine failed: %s", exc)
        return 1

    logging.info("Mock navigation completed. executed_steps=%d published_msgs=%d", steps, len(publisher.messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())