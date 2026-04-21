import math


class FakeRosOperator:
    def __init__(self, pose=None):
        self._pose = pose or {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.published: list[list[float]] = []

    def latest_odometry(self):
        return dict(self._pose)

    def robot_base_publish(self, values):
        self.published.append(list(values))


def test_navigate_runs_full_goal_routine(monkeypatch):
    from examples.piper_real import navigation_tool

    ros_operator = FakeRosOperator(pose={"x": 0.0, "y": 0.0, "yaw": 0.0})
    goal_calls: list[navigation_tool.NavigationGoal] = []
    sleep_calls: list[float] = []
    stop_calls: list[str] = []

    def fake_navigate_to_goal(_ros_operator, goal, _config):
        goal_calls.append(goal)
        return navigation_tool.CoordinateNavigationResult(
            ok=True,
            goal=goal,
            executed_steps=1,
            final_pose={"x": goal.x, "y": goal.y, "yaw": goal.yaw or 0.0},
        )

    monkeypatch.setattr(navigation_tool, "navigate_to_goal", fake_navigate_to_goal)
    monkeypatch.setattr(navigation_tool.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(
        navigation_tool.base_safety,
        "stop_base",
        lambda _ros: stop_calls.append("stop"),
    )

    result = navigation_tool.navigate("移动到桌边", ros_operator)

    assert result.ok is True
    assert result.prompt == "移动到桌边"
    assert result.routine_name == navigation_tool.DEFAULT_ROUTINE_NAME
    assert result.executed_steps == len(navigation_tool.DEFAULT_DEMO_GOAL_ROUTINE)
    assert result.error is None

    # Origin is (0,0,0) so absolute goals equal the body-frame offsets.
    expected = list(navigation_tool.DEFAULT_DEMO_GOAL_ROUTINE)
    assert len(goal_calls) == len(expected)
    for got, want in zip(goal_calls, expected):
        assert math.isclose(got.x, want.x, abs_tol=1e-9)
        assert math.isclose(got.y, want.y, abs_tol=1e-9)
        assert (got.yaw is None) == (want.yaw is None)
        if got.yaw is not None:
            assert math.isclose(got.yaw, want.yaw, abs_tol=1e-9)

    # inter-step sleep happens between goals (not after the last).
    assert sleep_calls.count(navigation_tool.INTER_STEP_SLEEP_S) == len(expected) - 1
    # final stop after successful routine.
    assert stop_calls == ["stop"]


def test_navigate_resolves_goals_relative_to_origin(monkeypatch):
    from examples.piper_real import navigation_tool

    ros_operator = FakeRosOperator(pose={"x": 1.0, "y": 2.0, "yaw": math.pi / 2.0})
    goal_calls: list[navigation_tool.NavigationGoal] = []

    def fake_navigate_to_goal(_ros_operator, goal, _config):
        goal_calls.append(goal)
        return navigation_tool.CoordinateNavigationResult(
            ok=True, goal=goal, executed_steps=1
        )

    monkeypatch.setattr(navigation_tool, "navigate_to_goal", fake_navigate_to_goal)
    monkeypatch.setattr(navigation_tool.time, "sleep", lambda _s: None)
    monkeypatch.setattr(navigation_tool.base_safety, "stop_base", lambda _ros: None)

    routine = (navigation_tool.NavigationGoal(x=0.5, y=0.0, yaw=0.0),)
    result = navigation_tool.navigate("rel", ros_operator, routine=routine)

    assert result.ok is True
    # body-frame +x=0.5 at yaw=pi/2 maps to odom (+0, +0.5) relative to origin.
    assert math.isclose(goal_calls[0].x, 1.0, abs_tol=1e-6)
    assert math.isclose(goal_calls[0].y, 2.5, abs_tol=1e-6)
    assert math.isclose(goal_calls[0].yaw, math.pi / 2.0, abs_tol=1e-6)


def test_navigate_dry_run_skips_motion_execution(monkeypatch):
    from examples.piper_real import navigation_tool

    def boom(*_a, **_k):
        raise AssertionError("navigate_to_goal must not run when dry_run=True")

    monkeypatch.setattr(navigation_tool, "navigate_to_goal", boom)

    result = navigation_tool.navigate("任意 prompt", None, dry_run=True)

    assert result.ok is True
    assert result.prompt == "任意 prompt"
    assert result.routine_name == navigation_tool.DEFAULT_ROUTINE_NAME
    assert result.executed_steps == 0
    assert result.error is None


def test_navigate_dry_run_handles_ros_operator_none(monkeypatch):
    from examples.piper_real import navigation_tool

    monkeypatch.setattr(
        navigation_tool,
        "navigate_to_goal",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    result = navigation_tool.navigate("prompt", None)

    assert result.ok is True
    assert result.executed_steps == 0


def test_navigate_stops_base_and_returns_failure_on_goal_error(monkeypatch):
    from examples.piper_real import navigation_tool

    ros_operator = FakeRosOperator()
    stop_calls: list[str] = []
    call_count = {"n": 0}

    def fake_navigate_to_goal(_ros_operator, goal, _config):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return navigation_tool.CoordinateNavigationResult(
                ok=False, goal=goal, executed_steps=0, error="boom"
            )
        return navigation_tool.CoordinateNavigationResult(
            ok=True, goal=goal, executed_steps=1
        )

    monkeypatch.setattr(navigation_tool, "navigate_to_goal", fake_navigate_to_goal)
    monkeypatch.setattr(navigation_tool.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        navigation_tool.base_safety,
        "stop_base",
        lambda _ros: stop_calls.append("stop"),
    )

    result = navigation_tool.navigate("去桌边", ros_operator)

    assert result.ok is False
    assert result.prompt == "去桌边"
    assert result.routine_name == navigation_tool.DEFAULT_ROUTINE_NAME
    assert result.executed_steps == 1  # only the first goal completed
    assert result.error == "boom"
    assert call_count["n"] == 2  # bailed after the failing goal
    assert stop_calls == ["stop"]


def test_navigate_fails_fast_when_odometry_missing(monkeypatch):
    from examples.piper_real import navigation_tool

    class NoOdomOperator:
        def latest_odometry(self):
            return None

        def robot_base_publish(self, values):
            self.last = list(values)

    ros_operator = NoOdomOperator()
    stop_calls: list[str] = []

    monkeypatch.setattr(
        navigation_tool,
        "navigate_to_goal",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setattr(navigation_tool.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        navigation_tool.base_safety,
        "stop_base",
        lambda _ros: stop_calls.append("stop"),
    )
    # Override _wait_for_odometry to short-circuit its internal sleep loop.
    monkeypatch.setattr(
        navigation_tool,
        "_wait_for_odometry",
        lambda _op, _timeout: None,
    )

    result = navigation_tool.navigate("prompt", ros_operator)

    assert result.ok is False
    assert result.executed_steps == 0
    assert result.error == "odometry unavailable"
    assert stop_calls == ["stop"]
