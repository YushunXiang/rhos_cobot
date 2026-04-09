import importlib
import sys
import types
from types import SimpleNamespace


def _load_tracer_demo(monkeypatch):
    fake_rospy = types.ModuleType("rospy")
    fake_rospy.init_node = lambda *args, **kwargs: None
    fake_rospy.Publisher = lambda *args, **kwargs: object()
    fake_rospy.sleep = lambda *args, **kwargs: None
    fake_rospy.ROSInterruptException = RuntimeError

    fake_geometry_msgs = types.ModuleType("geometry_msgs")
    fake_geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class FakeTwist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0)
            self.angular = SimpleNamespace(z=0.0)

    fake_geometry_msgs_msg.Twist = FakeTwist

    monkeypatch.setitem(sys.modules, "rospy", fake_rospy)
    monkeypatch.setitem(sys.modules, "geometry_msgs", fake_geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", fake_geometry_msgs_msg)
    sys.modules.pop("scripts.tracer.tracer_demo", None)
    return importlib.import_module("scripts.tracer.tracer_demo")


def test_tracer_demo_main_invokes_navigation_tool(monkeypatch):
    tracer_demo = _load_tracer_demo(monkeypatch)

    assert hasattr(tracer_demo, "main")
    assert hasattr(tracer_demo, "_build_ros_operator")

    recorded: dict[str, object] = {}

    class FakeRosOperator:
        def robot_base_publish(self, values):
            recorded.setdefault("published", []).append(tuple(values))

    def fake_build_ros_operator():
        recorded["built_operator"] = True
        return FakeRosOperator()

    def fake_navigate(prompt, ros_operator, *, dry_run=False):
        recorded["navigate_call"] = (prompt, dry_run, type(ros_operator).__name__)
        ros_operator.robot_base_publish([0.1, 0.0])
        return SimpleNamespace(
            ok=True,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=5,
            error=None,
        )

    monkeypatch.setattr(tracer_demo, "_build_ros_operator", fake_build_ros_operator)
    monkeypatch.setattr(tracer_demo.navigation_tool, "navigate", fake_navigate)

    exit_code = tracer_demo.main(["--prompt", "移动到桌边"])

    assert exit_code == 0
    assert recorded["built_operator"] is True
    assert recorded["navigate_call"] == ("移动到桌边", False, "FakeRosOperator")
    assert recorded["published"] == [(0.1, 0.0)]


def test_tracer_demo_main_returns_nonzero_on_navigation_failure(monkeypatch):
    tracer_demo = _load_tracer_demo(monkeypatch)

    assert hasattr(tracer_demo, "main")
    assert hasattr(tracer_demo, "_build_ros_operator")

    monkeypatch.setattr(tracer_demo, "_build_ros_operator", lambda: object())
    monkeypatch.setattr(
        tracer_demo.navigation_tool,
        "navigate",
        lambda prompt, ros_operator, *, dry_run=False: SimpleNamespace(
            ok=False,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=1,
            error="boom",
        ),
    )

    exit_code = tracer_demo.main(["--prompt", "移动到桌边"])

    assert exit_code == 1
