import math
from collections import deque
from dataclasses import dataclass

import pytest

# 按你的实际文件位置改这里的 import 路径
# 假设 tracer_demo_coordinates.py 在 scripts/tracer/
from scripts.tracer import tracer_demo_coordinates as demo


class FakePublisher:
    def __init__(self):
        self.published: list[tuple[float, float]] = []

    def publish(self, twist) -> None:
        self.published.append((float(twist.linear.x), float(twist.angular.z)))


class FakeTwist:
    # Fake massage
    class _Linear:
        def __init__(self):
            self.x = 0.0

    class _Angular:
        def __init__(self):
            self.z = 0.0

    def __init__(self):
        self.linear = self._Linear()
        self.angular = self._Angular()


@dataclass
class _Pose2D:
    x: float
    y: float
    yaw: float


class FakeRosOperator:
    """最小可用 ros_operator mock，实现被测函数需要的方法。"""

    def __init__(self, poses: list[_Pose2D]):
        self.publisher = FakePublisher()
        self.twist_type = FakeTwist
        self.odom_messages = deque(maxlen=1)
        self._poses = poses
        self._idx = 0

    def robot_base_publish(self, values) -> None:
        t = self.twist_type()
        t.linear.x = float(values[0])
        t.angular.z = float(values[1])
        self.publisher.publish(t)

    def latest_odometry(self):
        # 每次调用返回一帧“当前里程计”，模拟机器人逐步接近目标
        if not self._poses:
            return None
        pose = self._poses[min(self._idx, len(self._poses) - 1)]
        self._idx += 1
        return {"x": pose.x, "y": pose.y, "yaw": pose.yaw}


def test_compute_command_reached_without_yaw():
    # test if position reached but yaw is None, it should be considered reached
    goal = demo.NavigationGoal(x=1.0, y=1.0, yaw=None)
    cfg = demo.NavigationConfig(position_tolerance_m=0.2)

    current = {"x": 1.05, "y": 1.05, "yaw": 0.3}
    vx, wz, telemetry, reached = demo._compute_command(current, goal, cfg)  # noqa: SLF001

    assert reached is True
    assert vx == 0.0
    assert wz == 0.0
    assert telemetry["phase"] == "reached"


def test_compute_command_align_when_position_reached_but_yaw_not_reached():
    # test yaw not reached but position is reached, it should be in align phase
    goal = demo.NavigationGoal(x=0.0, y=0.0, yaw=1.0)
    cfg = demo.NavigationConfig(
        position_tolerance_m=0.2,
        yaw_tolerance_rad=0.05,
        angular_gain=1.0,
        max_angular_vel_rad_s=0.5,
    )

    current = {"x": 0.01, "y": -0.01, "yaw": 0.0}
    vx, wz, telemetry, reached = demo._compute_command(current, goal, cfg)  # noqa: SLF001

    assert reached is False
    assert vx == 0.0
    assert telemetry["phase"] == "align"
    assert 0.0 < wz <= 0.5


def test_navigate_to_goal_success_with_mock_odometry(monkeypatch):
    # 轨迹逐步接近 (1.0, 0.0)，最后进入容差
    poses = [
        _Pose2D(0.0, 0.0, 0.0),
        _Pose2D(0.3, 0.0, 0.0),
        _Pose2D(0.6, 0.0, 0.0),
        _Pose2D(0.85, 0.0, 0.0),
        _Pose2D(0.95, 0.0, 0.0),
        _Pose2D(1.0, 0.0, 0.0),
        _Pose2D(1.0, 0.0, 0.0),
    ]
    ros = FakeRosOperator(poses)

    # 避免真实 stop_base/enforce 限幅依赖，打桩成可控行为
    monkeypatch.setattr(demo.base_safety, "stop_base", lambda op: op.robot_base_publish([0.0, 0.0]))
    monkeypatch.setattr(
        demo.base_safety,
        "enforce_base_velocity_limits",
        lambda vx, wz, **kwargs: (vx, wz),
    )

    goal = demo.NavigationGoal(x=1.0, y=0.0, yaw=None)
    cfg = demo.NavigationConfig(
        control_hz=20.0,
        goal_timeout_s=5.0,
        odom_wait_timeout_s=0.5,
        goal_hold_seconds=0.0,  # 让测试更快收敛
        position_tolerance_m=0.08,
        heading_align_threshold_rad=0.5,
    )

    result = demo.navigate_to_goal(ros, goal, cfg)

    assert result.ok is True
    assert result.executed_steps > 0
    assert result.final_pose is not None
    # 至少发布过一次非零速度，说明确实执行了控制
    assert any(abs(vx) > 1e-6 or abs(wz) > 1e-6 for vx, wz in ros.publisher.published)


def test_navigate_to_goal_fails_when_odometry_unavailable(monkeypatch):
    class NoOdomOperator:
        def latest_odometry(self):
            return None

        def robot_base_publish(self, values):
            pass

    op = NoOdomOperator()

    called = {"stop": 0}

    def _fake_stop(_):
        called["stop"] += 1

    monkeypatch.setattr(demo.base_safety, "stop_base", _fake_stop)

    goal = demo.NavigationGoal(x=1.0, y=0.0, yaw=None)
    cfg = demo.NavigationConfig(odom_wait_timeout_s=0.1, goal_timeout_s=0.5, control_hz=10.0)

    result = demo.navigate_to_goal(op, goal, cfg)

    assert result.ok is False
    assert "odometry unavailable" in (result.error or "")
    assert called["stop"] >= 1


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi, math.pi),
    ],
)
def test_normalize_angle_basic(angle, expected):
    out = demo._normalize_angle(angle)  # noqa: SLF001
    assert math.isclose(out, expected, rel_tol=1e-6, abs_tol=1e-6)