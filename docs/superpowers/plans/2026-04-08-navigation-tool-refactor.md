# Navigation Tool Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace live `LLMNavigationPlanner` execution in `examples/piper_real/main.py` with a reusable `examples/piper_real/navigation_tool.py` module, keep `scripts/tracer/tracer_demo.py` as a thin demo/CLI wrapper, and preserve the current task decomposition plus manipulation Runtime flow.

**Architecture:** Add a small navigation module that owns the fixed TRACER routine and returns a structured `NavigationResult`. Refactor the live `main.py` path so navigate subtasks call the new tool and manipulate subtasks continue through the existing Runtime path. Convert `scripts/tracer/tracer_demo.py` into a wrapper that initializes ROS, builds a tiny `ros_operator` adapter, and delegates to the same navigation tool used by `main.py`.

**Tech Stack:** Python 3.11, ROS1 `rospy`, `tyro`, `pytest`, `monkeypatch`, `logging`

---

## File Structure

- Create: `examples/piper_real/navigation_tool.py`
  - Own the default TRACER routine, the step-execution loop, dry-run behavior, and `NavigationResult`.
- Modify: `examples/piper_real/main.py`
  - Remove live use of `LLMNavigationPlanner`, remove dead `BASE_ROUTINE` / `fixed_navigation`, and route navigate subtasks through `navigation_tool.navigate(...)`.
- Create: `tests/test_navigation_tool.py`
  - Unit-test the tool contract: default routine selection, dry-run behavior, and failure handling.
- Create: `scripts/tracer/__init__.py`
  - Make `scripts.tracer.tracer_demo` importable by pytest.
- Modify: `scripts/tracer/tracer_demo.py`
  - Replace the embedded routine with a thin CLI wrapper and ROS publisher adapter.
- Delete: `scripts/tracer/test_tracer_demo.py`
  - Remove the manual script-style test once pytest coverage exists in `tests/`.
- Create: `tests/test_tracer_demo.py`
  - Unit-test the CLI wrapper behavior without requiring a live ROS environment.
- Create: `tests/test_main_navigation_flow.py`
  - Lock in live-path behavior: navigate calls the tool, failure aborts manipulate, and `navigation_only` still skips manipulation.
- Modify: `examples/piper_real/README.md`
  - Update the architecture, planner-service contract, and validation notes to describe decomposition plus tool-based navigation.
- Modify: `docs/deploy.md`
  - Replace live `LLMNavigationPlanner` references with the navigation tool architecture and the new decomposition JSON contract.

## Task 1: Add the reusable navigation tool

**Files:**
- Create: `examples/piper_real/navigation_tool.py`
- Create: `tests/test_navigation_tool.py`

- [ ] **Step 1: Write the failing tests for the navigation tool**

```python
# tests/test_navigation_tool.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_navigation_tool.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples.piper_real.navigation_tool'`.

- [ ] **Step 3: Write the minimal navigation tool implementation**

```python
# examples/piper_real/navigation_tool.py
import dataclasses
import logging
import time
from typing import Any

from examples.piper_real import base_safety

DEFAULT_CONTROL_HZ = 10.0
INTER_STEP_SLEEP_S = 1.0
DEFAULT_ROUTINE_NAME = "default_demo"
DEFAULT_DEMO_ROUTINE: tuple[tuple[float, float, float], ...] = (
    (-0.2, 0.0, 1.0),
    (0.0, 0.2, 9.0),
    (0.2, 0.0, 1.5),
    (0.0, -0.2, 9.0),
    (0.1, 0.0, 2.0),
)


@dataclasses.dataclass
class NavigationResult:
    ok: bool
    prompt: str
    routine_name: str
    executed_steps: int
    error: str | None = None


def _execute_step(
    ros_operator: Any,
    step: tuple[float, float, float],
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
) -> None:
    linear_x, angular_z, duration = step
    start = time.monotonic()
    period = 1.0 / control_hz
    next_tick = start

    while time.monotonic() - start < duration:
        ros_operator.robot_base_publish([linear_x, angular_z])
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()

    base_safety.stop_base(ros_operator)


def navigate(
    prompt: str,
    ros_operator: Any | None,
    *,
    dry_run: bool = False,
) -> NavigationResult:
    logging.info(
        "Navigation tool invoked: prompt=%s routine=%s dry_run=%s",
        prompt,
        DEFAULT_ROUTINE_NAME,
        dry_run,
    )

    if dry_run:
        return NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=0,
        )

    if ros_operator is None:
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=0,
            error="ros_operator is required when dry_run is False",
        )

    executed_steps = 0
    try:
        for idx, step in enumerate(DEFAULT_DEMO_ROUTINE, start=1):
            logging.info(
                "Navigation step %d/%d: linear_x=%s angular_z=%s duration=%s",
                idx,
                len(DEFAULT_DEMO_ROUTINE),
                step[0],
                step[1],
                step[2],
            )
            _execute_step(ros_operator, step)
            executed_steps = idx
            if idx < len(DEFAULT_DEMO_ROUTINE):
                time.sleep(INTER_STEP_SLEEP_S)
    except Exception as exc:  # noqa: BLE001
        base_safety.stop_base(ros_operator)
        logging.error("Navigation tool failed after %d steps: %s", executed_steps, exc)
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=executed_steps,
            error=str(exc),
        )

    base_safety.stop_base(ros_operator)
    return NavigationResult(
        ok=True,
        prompt=prompt,
        routine_name=DEFAULT_ROUTINE_NAME,
        executed_steps=executed_steps,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_navigation_tool.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_navigation_tool.py examples/piper_real/navigation_tool.py
git commit -m "feat: add reusable navigation tool"
```

## Task 2: Convert `tracer_demo.py` into a thin CLI wrapper

**Files:**
- Create: `scripts/tracer/__init__.py`
- Modify: `scripts/tracer/tracer_demo.py:1-61`
- Delete: `scripts/tracer/test_tracer_demo.py`
- Create: `tests/test_tracer_demo.py`

- [ ] **Step 1: Write the failing wrapper tests**

```python
# tests/test_tracer_demo.py
from types import SimpleNamespace


def test_tracer_demo_main_invokes_navigation_tool(monkeypatch):
    import scripts.tracer.tracer_demo as tracer_demo

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
    import scripts.tracer.tracer_demo as tracer_demo

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_tracer_demo.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.tracer'` or `AttributeError` because the wrapper helpers do not exist yet.

- [ ] **Step 3: Replace the embedded demo routine with the wrapper implementation**

```python
# scripts/tracer/__init__.py
"""TRACER demo scripts."""
```

```python
# scripts/tracer/tracer_demo.py
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
```

Delete the old manual test file:

```bash
rm scripts/tracer/test_tracer_demo.py
```

- [ ] **Step 4: Run the wrapper tests**

Run:

```bash
pytest tests/test_tracer_demo.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/tracer/__init__.py scripts/tracer/tracer_demo.py tests/test_tracer_demo.py
git rm scripts/tracer/test_tracer_demo.py
git commit -m "refactor: wrap tracer demo around navigation tool"
```

## Task 3: Route live navigate subtasks through the navigation tool

**Files:**
- Modify: `examples/piper_real/main.py:18-45`
- Modify: `examples/piper_real/main.py:590-771`
- Create: `tests/test_main_navigation_flow.py`

- [ ] **Step 1: Write the failing live-path integration tests**

```python
# tests/test_main_navigation_flow.py
from types import SimpleNamespace


def test_main_calls_navigation_tool_before_manipulation(monkeypatch):
    from openpi_client.runtime import runtime as runtime_mod
    from examples.piper_real import base_safety as base_safety_mod
    from examples.piper_real import env as env_mod
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"runtime_runs": 0, "stop_calls": 0}

    class FakeEnvironment:
        def __init__(self, reset_position, prompt):
            recorded["environment_init"] = (reset_position, prompt)
            self.ros_operator = SimpleNamespace(name="live-ros")

        def set_prompt(self, prompt):
            recorded.setdefault("manipulate_prompts", []).append(prompt)

    class FakeTaskDecomposer:
        def __init__(self, _config):
            pass

        def decompose(self, _prompt):
            return [
                task_decomposer_mod.Subtask(type="navigate", prompt="move to table"),
                task_decomposer_mod.Subtask(type="manipulate", prompt="pick cup"),
            ]

    class FakeWebsocketClientPolicy:
        def __init__(self, host, port):
            recorded["server"] = (host, port)

        def get_server_metadata(self):
            return {"reset_pose": [0.0] * 14}

    class FakeRuntime:
        def __init__(self, environment, agent, subscribers, max_hz, num_episodes, max_episode_steps):
            recorded["runtime_init"] = {
                "environment": environment,
                "agent": agent,
                "subscribers": subscribers,
                "max_hz": max_hz,
                "num_episodes": num_episodes,
                "max_episode_steps": max_episode_steps,
            }

        def run(self):
            recorded["runtime_runs"] += 1

    def fake_navigate(prompt, ros_operator, *, dry_run=False):
        recorded.setdefault("navigate_calls", []).append((prompt, ros_operator, dry_run))
        return navigation_tool_mod.NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=5,
        )

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(env_mod, "PiperRealEnvironment", FakeEnvironment)
    monkeypatch.setattr(runtime_mod, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        main_module._websocket_client_policy,
        "WebsocketClientPolicy",
        FakeWebsocketClientPolicy,
    )
    monkeypatch.setattr(main_module._policy_agent, "PolicyAgent", lambda policy: ("agent", policy))
    monkeypatch.setattr(
        main_module.action_chunk_broker,
        "ActionChunkBroker",
        lambda policy, action_horizon: ("broker", action_horizon),
    )
    monkeypatch.setattr(main_module, "_run_required_server_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(base_safety_mod, "confirm_base_motion_safety", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        base_safety_mod,
        "stop_base",
        lambda _ros_operator: recorded.__setitem__("stop_calls", recorded["stop_calls"] + 1),
    )
    monkeypatch.setattr(navigation_tool_mod, "navigate", fake_navigate)

    args = main_module.Args(
        use_llm_planner=True,
        use_robot_base=True,
        prompt="move to the table and pick the cup",
        skip_server_checks=True,
    )
    monkeypatch.setattr(
        args.planner,
        "validate_motion_limits",
        lambda: (_ for _ in ()).throw(AssertionError("validate_motion_limits should not run")),
    )

    main_module.main(args)

    assert recorded["navigate_calls"][0][0] == "move to table"
    assert recorded["navigate_calls"][0][2] is False
    assert recorded["manipulate_prompts"] == ["pick cup"]
    assert recorded["runtime_runs"] == 1
    assert recorded["stop_calls"] == 1


def test_main_aborts_manipulation_when_navigation_tool_fails(monkeypatch):
    from openpi_client.runtime import runtime as runtime_mod
    from examples.piper_real import base_safety as base_safety_mod
    from examples.piper_real import env as env_mod
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"runtime_runs": 0, "stop_calls": 0}

    class FakeEnvironment:
        def __init__(self, reset_position, prompt):
            self.ros_operator = SimpleNamespace(name="live-ros")

        def set_prompt(self, prompt):
            recorded.setdefault("manipulate_prompts", []).append(prompt)

    class FakeTaskDecomposer:
        def __init__(self, _config):
            pass

        def decompose(self, _prompt):
            return [
                task_decomposer_mod.Subtask(type="navigate", prompt="move to table"),
                task_decomposer_mod.Subtask(type="manipulate", prompt="pick cup"),
            ]

    class FakeWebsocketClientPolicy:
        def __init__(self, host, port):
            pass

        def get_server_metadata(self):
            return {"reset_pose": [0.0] * 14}

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            recorded["runtime_runs"] += 1

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(env_mod, "PiperRealEnvironment", FakeEnvironment)
    monkeypatch.setattr(runtime_mod, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        main_module._websocket_client_policy,
        "WebsocketClientPolicy",
        FakeWebsocketClientPolicy,
    )
    monkeypatch.setattr(main_module._policy_agent, "PolicyAgent", lambda policy: ("agent", policy))
    monkeypatch.setattr(
        main_module.action_chunk_broker,
        "ActionChunkBroker",
        lambda policy, action_horizon: ("broker", action_horizon),
    )
    monkeypatch.setattr(main_module, "_run_required_server_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(base_safety_mod, "confirm_base_motion_safety", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        base_safety_mod,
        "stop_base",
        lambda _ros_operator: recorded.__setitem__("stop_calls", recorded["stop_calls"] + 1),
    )
    monkeypatch.setattr(
        navigation_tool_mod,
        "navigate",
        lambda prompt, ros_operator, *, dry_run=False: navigation_tool_mod.NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=1,
            error="boom",
        ),
    )

    main_module.main(
        main_module.Args(
            use_llm_planner=True,
            use_robot_base=True,
            prompt="move to the table and pick the cup",
            skip_server_checks=True,
        )
    )

    assert recorded["runtime_runs"] == 0
    assert recorded["stop_calls"] == 1
    assert "manipulate_prompts" not in recorded


def test_main_navigation_only_uses_dry_run_without_runtime(monkeypatch):
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {}

    class FakeTaskDecomposer:
        def __init__(self, _config):
            pass

        def decompose(self, _prompt):
            return [
                task_decomposer_mod.Subtask(type="navigate", prompt="move to table"),
                task_decomposer_mod.Subtask(type="manipulate", prompt="pick cup"),
            ]

    def fake_navigate(prompt, ros_operator, *, dry_run=False):
        recorded["navigate_call"] = (prompt, ros_operator, dry_run)
        return navigation_tool_mod.NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=0,
        )

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(main_module, "_run_required_server_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(navigation_tool_mod, "navigate", fake_navigate)

    main_module.main(
        main_module.Args(
            use_llm_planner=True,
            navigation_only=True,
            prompt="move to the table",
            skip_server_checks=True,
        )
    )

    assert recorded["navigate_call"] == ("move to table", None, True)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_main_navigation_flow.py -v
```

Expected: FAIL because `examples/piper_real/main.py` still constructs `LLMNavigationPlanner`, still honors `fixed_navigation`, and does not call `navigation_tool.navigate(...)`.

- [ ] **Step 3: Refactor the live `main.py` path to call the navigation tool**

```python
# examples/piper_real/main.py
DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_REPLAY_MANIPULATE_MAX_STEPS = 64
DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS = 16
_VALID_REPLAY_MODES = {"policy", "planner", "hybrid"}


@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"
    port: int = 9000
    action_horizon: int = 16
    num_episodes: int = 1
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    save_log: bool = False
    prompt: str = ""
    replay_dataset: str = ""
    replay_mode: str = "policy"
    replay_manipulate_max_steps: int = DEFAULT_REPLAY_MANIPULATE_MAX_STEPS
    replay_manipulate_replan_interval_steps: int = DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS
    use_llm_planner: bool = False
    use_robot_base: bool = False
    navigation_only: bool = False
    skip_server_checks: bool = False
    server_check_timeout_sec: float = 5.0
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)
```

```python
# examples/piper_real/main.py inside main()
from examples.piper_real import base_safety as _base_safety
from examples.piper_real import env as _env
from examples.piper_real import logger as _logger
from examples.piper_real import navigation_tool as _navigation_tool
from examples.piper_real import task_decomposer as _task_decomposer
from openpi_client.runtime import runtime as _runtime

# ── Two-layer task decomposition + tool-based navigation ─────────────
args.planner.validate_service_config()

if not _run_required_server_checks(args, needs_planner=True):
    return

decomposer = _task_decomposer.TaskDecomposer(args.planner)
try:
    subtask_list = decomposer.decompose(prompt)
except _task_decomposer.DecompositionError as exc:
    logging.error("Task decomposition failed: %s", exc)
    return

has_navigate = any(s.type == "navigate" for s in subtask_list)
has_manipulate = any(s.type == "manipulate" for s in subtask_list)
needs_server = has_manipulate and not args.navigation_only
needs_ros_environment = needs_server or (args.use_robot_base and has_navigate)

if args.use_robot_base and has_navigate:
    if not _base_safety.confirm_base_motion_safety(
        prompt,
        use_llm_planner=True,
        use_robot_base=False,
    ):
        logging.error("Base motion aborted before execution.")
        return

ws_client_policy = None
metadata = {}
if needs_server:
    if not _run_required_server_checks(args, needs_pi0=True):
        return
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

environment = None
if needs_ros_environment:
    if args.save_log and needs_server:
        _logger.InputJointStateLogger()
        _logger.OutputJointStateLogger()

    environment = _env.PiperRealEnvironment(
        reset_position=metadata.get("reset_pose") if needs_server else None,
        prompt=prompt,
    )

try:
    for idx, subtask in enumerate(subtask_list):
        logging.info(
            "Executing subtask %d/%d [%s]: %s",
            idx + 1,
            len(subtask_list),
            subtask.type,
            subtask.prompt,
        )

        if subtask.type == "navigate":
            ros_operator = None if environment is None else environment.ros_operator
            result = _navigation_tool.navigate(
                subtask.prompt,
                ros_operator,
                dry_run=not args.use_robot_base,
            )
            if not result.ok:
                logging.error(
                    "Navigation failed at subtask %d/%d: %s",
                    idx + 1,
                    len(subtask_list),
                    result.error or "unknown error",
                )
                return
            logging.info(
                "Navigate subtask %d/%d succeeded via routine %s.",
                idx + 1,
                len(subtask_list),
                result.routine_name,
            )
            continue

        if args.navigation_only:
            logging.info("Manipulate (skipped): %s", subtask.prompt)
            continue

        assert ws_client_policy is not None, "manipulate subtask requires server connection"
        assert environment is not None, "manipulate subtask requires a live environment"
        environment.set_prompt(subtask.prompt)
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=50,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
        logging.info("Manipulate subtask %d/%d completed.", idx + 1, len(subtask_list))
finally:
    if environment is not None and args.use_robot_base:
        _base_safety.stop_base(environment.ros_operator)
```

- [ ] **Step 4: Run the live-path tests**

Run:

```bash
pytest tests/test_main_navigation_flow.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add examples/piper_real/main.py tests/test_main_navigation_flow.py
git commit -m "refactor: route live navigation through tool"
```

## Task 4: Update docs and remove stale live-navigation references

**Files:**
- Modify: `examples/piper_real/README.md`
- Modify: `docs/deploy.md`

- [ ] **Step 1: Update `examples/piper_real/README.md` to describe decomposition plus tool-based navigation**

Replace the overview and planner-contract sections with:

```markdown
The flow is now decomposition-first:

1. Build `PiperRealEnvironment` and connect to the OpenPI policy server only when a manipulate subtask exists.
2. If `--use-llm-planner` is enabled and `--prompt` is non-empty, send the full task prompt to the planner service and get back ordered `navigate` / `manipulate` subtasks.
3. Execute each `navigate` subtask by calling `examples.piper_real.navigation_tool.navigate(...)`, which runs the fixed `default_demo` TRACER routine in v1.
4. Execute each `manipulate` subtask with the existing OpenPI Runtime path.
```

````markdown
The planner must return this shape:

```json
{
  "subtasks": [
    {"type": "navigate", "prompt": "move to the table"},
    {"type": "manipulate", "prompt": "pick up the red cup"}
  ]
}
```

Runtime behavior:

- A safety warning is shown before navigation starts when `--use-robot-base` is enabled.
- The operator must type `yes` to allow base motion.
- Each navigate subtask invokes the shared navigation tool.
- In v1, every navigate prompt runs the same fixed `default_demo` routine.
- Manipulation starts only after the navigate subtask returns success.
```
````

- [ ] **Step 2: Update `docs/deploy.md` to remove live `LLMNavigationPlanner` wording**

Replace the live-architecture section with:

````markdown
## 5. LLM 两层任务架构（`--use-llm-planner`）

live 路径现在分为两个阶段：

1. **TaskDecomposer**：把完整任务拆解成有序的 `navigate` / `manipulate` subtask 列表。
2. **navigation_tool + Runtime**：
   - `navigate` subtask 调用 `examples/piper_real/navigation_tool.py`
   - `manipulate` subtask 继续走现有 OpenPI Runtime

`navigate` 在 v1 不做 prompt-to-routine 路由；任何 navigate prompt 都执行固定的 `default_demo` TRACER routine。
```
````

Also replace the planner-response example with:

````markdown
返回 JSON only：

```json
{
  "subtasks": [
    {"type": "navigate", "prompt": "move to the table"},
    {"type": "manipulate", "prompt": "pick up the red cup"}
  ]
}
```
````

And replace any sentence that says base motion is driven by `LLMNavigationPlanner` with:

```markdown
底盘移动只发生在 navigate subtask 中，并由 `examples/piper_real/navigation_tool.py` 执行。
```

- [ ] **Step 3: Verify the docs no longer describe the old live planner loop**

Run:

```bash
rg -n 'LLMNavigationPlanner|planner returns `stop`|"action": "move"|"action": "stop"' examples/piper_real/README.md docs/deploy.md
pytest tests/test_navigation_tool.py tests/test_tracer_demo.py tests/test_main_navigation_flow.py tests/test_llm_response_parsing.py tests/test_replay_env.py -v
```

Expected:

- `rg` returns no matches for stale live-path wording in `examples/piper_real/README.md` and `docs/deploy.md`
- `pytest` reports all selected tests passing

- [ ] **Step 4: Commit**

```bash
git add examples/piper_real/README.md docs/deploy.md
git commit -m "docs: update navigation tool workflow"
```

## Verification Commands

Run these after Task 4 before handing the branch back:

```bash
pytest tests/test_navigation_tool.py tests/test_tracer_demo.py tests/test_main_navigation_flow.py tests/test_llm_response_parsing.py tests/test_replay_env.py -v
ruff check examples/piper_real/navigation_tool.py examples/piper_real/main.py scripts/tracer/tracer_demo.py tests/test_navigation_tool.py tests/test_tracer_demo.py tests/test_main_navigation_flow.py
git status --short
```

Expected:

- All selected pytest cases pass.
- `ruff check` reports no violations in the changed files.
- `git status --short` shows only the files intentionally changed by this plan.
