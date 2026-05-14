import importlib
import sys
import types
from collections import deque
from types import SimpleNamespace


class _Stamp:
    def __init__(self, value: float):
        self._value = value

    def to_sec(self) -> float:
        return self._value


def _install_data_collection_stubs(monkeypatch, *, now: float = 11.0) -> None:
    rospy = types.SimpleNamespace(
        Time=types.SimpleNamespace(now=lambda: _Stamp(now)),
        init_node=lambda *args, **kwargs: None,
        Subscriber=lambda *args, **kwargs: None,
        Rate=lambda *_args, **_kwargs: types.SimpleNamespace(sleep=lambda: None),
        is_shutdown=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = type("JointState", (), {})
    sensor_msgs_msg.Image = type("Image", (), {})
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msgs_msg)

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    monkeypatch.setitem(sys.modules, "nav_msgs", nav_msgs)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msgs_msg)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = lambda: types.SimpleNamespace(
        imgmsg_to_cv2=lambda msg, _encoding: getattr(msg, "payload", msg)
    )
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)

    cv2 = types.SimpleNamespace(
        BORDER_CONSTANT=0,
        COLOR_BGR2RGB=0,
        copyMakeBorder=lambda img, *_args, **_kwargs: img,
        cvtColor=lambda img, _code: img,
        destroyAllWindows=lambda: None,
        imencode=lambda _ext, _img: (True, b"encoded"),
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=object))
    monkeypatch.setitem(sys.modules, "keyboard", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "dm_env",
        types.SimpleNamespace(
            StepType=types.SimpleNamespace(FIRST="FIRST", MID="MID"),
            TimeStep=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ),
    )


def _load_module(monkeypatch, *, now: float = 11.0):
    _install_data_collection_stubs(monkeypatch, now=now)
    sys.modules.pop("rhos_cobot.data_collection", None)
    return importlib.import_module("rhos_cobot.data_collection")


def _new_operator(module):
    operator = module.RosOperator.__new__(module.RosOperator)
    operator.args = SimpleNamespace(use_depth_image=False, use_robot_base=False)
    operator.bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda msg, _encoding: getattr(msg, "payload", msg)
    )
    for name in (
        "img_left_deque",
        "img_right_deque",
        "img_front_deque",
        "img_left_depth_deque",
        "img_right_depth_deque",
        "img_front_depth_deque",
        "master_arm_left_deque",
        "master_arm_right_deque",
        "puppet_arm_left_deque",
        "puppet_arm_right_deque",
        "puppet_eef_left_deque",
        "puppet_eef_right_deque",
        "robot_base_deque",
    ):
        setattr(operator, name, deque())
    return operator


def _msg(stamp: float, *, payload=None):
    return SimpleNamespace(header=SimpleNamespace(stamp=_Stamp(stamp)), payload=payload)


def _joint(stamp: float):
    msg = _msg(stamp)
    msg.position = [0.0] * 7
    msg.velocity = [0.0] * 7
    msg.effort = [0.0] * 7
    return msg


def _pose(stamp: float):
    msg = _msg(stamp)
    msg.pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return msg


def test_zero_stamped_robot_topics_are_timestamped_on_receipt(monkeypatch):
    module = _load_module(monkeypatch, now=11.0)
    operator = _new_operator(module)
    operator.img_left_deque.append(_msg(10.0, payload="left"))
    operator.img_right_deque.append(_msg(10.0, payload="right"))
    operator.img_front_deque.append(_msg(10.0, payload="front"))

    operator.master_arm_left_callback(_joint(0.0))
    operator.master_arm_right_callback(_joint(0.0))
    operator.puppet_arm_left_callback(_joint(0.0))
    operator.puppet_arm_right_callback(_joint(0.0))
    operator.puppet_eef_left_callback(_pose(0.0))
    operator.puppet_eef_right_callback(_pose(0.0))

    result = operator.get_frame()

    assert result is not False
    assert result[8].header.stamp.to_sec() == 11.0
    assert result[11].header.stamp.to_sec() == 11.0


def test_sync_failure_reason_reports_empty_and_stale_sources(monkeypatch):
    module = _load_module(monkeypatch)
    operator = _new_operator(module)
    operator.img_left_deque.append(_msg(10.0, payload="left"))
    operator.img_right_deque.append(_msg(10.0, payload="right"))
    operator.img_front_deque.append(_msg(10.0, payload="front"))
    operator.puppet_arm_left_deque.append(_joint(8.0))

    reason = operator._sync_failure_reason()

    assert "frame_time=10.000" in reason
    assert "master_arm_left: empty" in reason
    assert "puppet_arm_left: stale" in reason


def test_collect_parser_accepts_false_bool_strings(monkeypatch):
    _load_module(monkeypatch)
    sys.modules.pop("scripts.collect.collect_data_eef_qpos", None)
    module = importlib.import_module("scripts.collect.collect_data_eef_qpos")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_data_eef_qpos",
            "--use_depth_image",
            "False",
            "--use_robot_base",
            "false",
        ],
    )

    args = module.get_arguments()

    assert args.use_depth_image is False
    assert args.use_robot_base is False
