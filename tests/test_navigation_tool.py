def test_navigate_uses_default_demo_for_any_prompt(monkeypatch):
    from examples.piper_real import navigation_tool

    step_calls: list[tuple[float, float, float]] = []
    stop_calls: list[str] = []

    class FakeRosOperator:
        def robot_base_publish(self, _values):
            raise AssertionError("robot_base_publish should not be called directly in this test")

    def fake_execute_step(ros_operator, step, *, control_hz=navigation_tool.DEFAULT_CONTROL_HZ):
        assert isinstance(ros_operator, FakeRosOperator)
        step_calls.append(step)

    monkeypatch.setattr(navigation_tool, "_execute_step", fake_execute_step)
    monkeypatch.setattr(
        navigation_tool.base_safety,
        "stop_base",
        lambda _ros_operator: stop_calls.append("stop"),
    )

    result = navigation_tool.navigate("移动到桌边", FakeRosOperator())

    assert result.ok is True
    assert result.prompt == "移动到桌边"
    assert result.routine_name == "default_demo"
    assert result.executed_steps == len(navigation_tool.DEFAULT_DEMO_ROUTINE)
    assert result.error is None
    assert step_calls == list(navigation_tool.DEFAULT_DEMO_ROUTINE)
    assert stop_calls == ["stop"]


def test_navigate_dry_run_skips_motion_execution(monkeypatch):
    from examples.piper_real import navigation_tool

    monkeypatch.setattr(
        navigation_tool,
        "_execute_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("_execute_step should not run when dry_run=True")
        ),
    )

    result = navigation_tool.navigate("任意 prompt", None, dry_run=True)

    assert result.ok is True
    assert result.prompt == "任意 prompt"
    assert result.routine_name == "default_demo"
    assert result.executed_steps == 0
    assert result.error is None


def test_navigate_stops_base_and_returns_failure_on_step_error(monkeypatch):
    from examples.piper_real import navigation_tool

    call_state = {"count": 0, "stops": 0}

    class FakeRosOperator:
        def robot_base_publish(self, _values):
            pass

    def fake_execute_step(_ros_operator, _step, *, control_hz=navigation_tool.DEFAULT_CONTROL_HZ):
        if call_state["count"] == 1:
            raise RuntimeError("boom")
        call_state["count"] += 1

    monkeypatch.setattr(navigation_tool, "_execute_step", fake_execute_step)
    monkeypatch.setattr(
        navigation_tool.base_safety,
        "stop_base",
        lambda _ros_operator: call_state.__setitem__("stops", call_state["stops"] + 1),
    )

    result = navigation_tool.navigate("去桌边", FakeRosOperator())

    assert result.ok is False
    assert result.prompt == "去桌边"
    assert result.routine_name == "default_demo"
    assert result.executed_steps == 1
    assert result.error == "boom"
    assert call_state["stops"] == 1
