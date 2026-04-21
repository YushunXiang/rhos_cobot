import sys
import types
from types import SimpleNamespace


def _install_live_main_fakes(monkeypatch, recorded, *, planner_run_routine_result=True):
    fake_env_module = types.ModuleType("examples.piper_real.env")
    fake_logger_module = types.ModuleType("examples.piper_real.logger")
    fake_llm_planner_module = types.ModuleType("examples.piper_real.llm_planner")
    fake_replay_manipulation_planner_module = types.ModuleType(
        "examples.piper_real.replay_manipulation_planner"
    )

    class FakeEnvironment:
        def __init__(
            self,
            reset_position,
            prompt,
            robot_base_topic="/odom_raw",
            robot_base_cmd_topic="/cmd_vel",
        ):
            recorded.setdefault("environment_init", []).append(
                {
                    "reset_position": reset_position,
                    "prompt": prompt,
                    "robot_base_topic": robot_base_topic,
                    "robot_base_cmd_topic": robot_base_cmd_topic,
                }
            )
            self.ros_operator = SimpleNamespace(name="live-ros")

        def reset(self):
            recorded.setdefault("events", []).append("environment_reset")
            recorded["environment_reset_calls"] = recorded.get("environment_reset_calls", 0) + 1

        def set_prompt(self, prompt):
            recorded.setdefault("manipulate_prompts", []).append(prompt)

        def close(self):
            recorded["environment_close_calls"] = recorded.get("environment_close_calls", 0) + 1

    class FakeLLMNavigationPlanner:
        def __init__(self, ros_operator, _config):
            recorded.setdefault("planner_inits", []).append(ros_operator)

        def run_routine(self, routine):
            recorded.setdefault("planner_run_routine", []).append(routine)
            return planner_run_routine_result

        def run(self, task_prompt):
            recorded.setdefault("planner_run", []).append(task_prompt)
            return planner_run_routine_result

    class FakeReplayManipulationPromptPlanner:
        def __init__(self, environment, config, task_memory_runtime=None):
            recorded.setdefault("manipulation_planner_inits", []).append(
                (environment, config, task_memory_runtime)
            )

    fake_env_module.PiperRealEnvironment = FakeEnvironment
    fake_logger_module.InputJointStateLogger = lambda: recorded.setdefault("input_logger", 0) or None
    fake_logger_module.OutputJointStateLogger = lambda: recorded.setdefault("output_logger", 0) or None
    fake_llm_planner_module.LLMNavigationPlanner = FakeLLMNavigationPlanner
    fake_replay_manipulation_planner_module.ReplayManipulationPromptPlanner = (
        FakeReplayManipulationPromptPlanner
    )

    monkeypatch.setitem(sys.modules, "examples.piper_real.env", fake_env_module)
    monkeypatch.setitem(sys.modules, "examples.piper_real.logger", fake_logger_module)
    monkeypatch.setitem(sys.modules, "examples.piper_real.llm_planner", fake_llm_planner_module)
    monkeypatch.setitem(
        sys.modules,
        "examples.piper_real.replay_manipulation_planner",
        fake_replay_manipulation_planner_module,
    )

    import examples.piper_real as piper_real_package

    monkeypatch.setattr(piper_real_package, "env", fake_env_module, raising=False)
    monkeypatch.setattr(piper_real_package, "logger", fake_logger_module, raising=False)
    monkeypatch.setattr(
        piper_real_package,
        "replay_manipulation_planner",
        fake_replay_manipulation_planner_module,
        raising=False,
    )


def test_main_calls_navigation_tool_before_manipulation(monkeypatch):
    from openpi_client.runtime import runtime as runtime_mod
    from examples.piper_real import base_safety as base_safety_mod
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"runtime_runs": 0, "stop_calls": 0}

    _install_live_main_fakes(monkeypatch, recorded, planner_run_routine_result=True)

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
    class FakePolicyAgent:
        policy_metadata = {"reset_pose": [0.0] * 14}

        def reset(self):
            recorded.setdefault("events", []).append("agent_reset")
            recorded["agent_reset_calls"] = recorded.get("agent_reset_calls", 0) + 1

    def fake_run_manipulation_subtask(*_args, subtask_prompt, **_kwargs):
        recorded.setdefault("events", []).append("manipulate")
        recorded.setdefault("manipulation_calls", []).append(subtask_prompt)
        return {"executed_steps": 3, "prompt_queries": 1, "completed": True}

    monkeypatch.setattr(main_module, "_create_policy_agent", lambda _args: FakePolicyAgent())
    monkeypatch.setattr(main_module, "_build_ordered_task_memory_runtime", lambda *_args: None)
    monkeypatch.setattr(main_module, "_run_manipulation_subtask", fake_run_manipulation_subtask)
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
        robot_base_topic="/odom",
        robot_base_cmd_topic="/cmd_vel",
        prompt="move to the table and pick the cup",
        skip_server_checks=True,
    )
    monkeypatch.setattr(args.planner, "validate_service_config", lambda: None)
    monkeypatch.setattr(args.planner, "validate_motion_limits", lambda: None)

    main_module.main(args)

    assert recorded["environment_reset_calls"] == 1
    assert recorded["environment_init"][0]["robot_base_topic"] == "/odom"
    assert recorded["environment_init"][0]["robot_base_cmd_topic"] == "/cmd_vel"
    assert recorded["agent_reset_calls"] == 1
    assert recorded["navigate_calls"][0][0] == "move to table"
    assert recorded["navigate_calls"][0][2] is False
    assert recorded["manipulate_prompts"] == ["pick cup"]
    assert recorded["manipulation_calls"] == ["pick cup"]
    assert recorded["events"] == ["environment_reset", "agent_reset", "manipulate"]
    assert recorded["stop_calls"] == 1
    assert recorded["environment_close_calls"] == 1


def test_main_aborts_manipulation_when_navigation_tool_fails(monkeypatch):
    from openpi_client.runtime import runtime as runtime_mod
    from examples.piper_real import base_safety as base_safety_mod
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"runtime_runs": 0, "stop_calls": 0}

    _install_live_main_fakes(monkeypatch, recorded, planner_run_routine_result=False)

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
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            recorded["runtime_runs"] += 1

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
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
    class FakePolicyAgent:
        policy_metadata = {"reset_pose": [0.0] * 14}

        def reset(self):
            recorded.setdefault("events", []).append("agent_reset")
            recorded["agent_reset_calls"] = recorded.get("agent_reset_calls", 0) + 1

    def fake_run_manipulation_subtask(*_args, **_kwargs):
        recorded["runtime_runs"] += 1
        return {"executed_steps": 1, "prompt_queries": 0, "completed": True}

    monkeypatch.setattr(main_module, "_create_policy_agent", lambda _args: FakePolicyAgent())
    monkeypatch.setattr(main_module, "_build_ordered_task_memory_runtime", lambda *_args: None)
    monkeypatch.setattr(main_module, "_run_manipulation_subtask", fake_run_manipulation_subtask)
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

    args = main_module.Args(
        use_llm_planner=True,
        use_robot_base=True,
        prompt="move to the table and pick the cup",
        skip_server_checks=True,
    )
    monkeypatch.setattr(args.planner, "validate_service_config", lambda: None)
    monkeypatch.setattr(args.planner, "validate_motion_limits", lambda: None)

    main_module.main(args)

    assert recorded["environment_reset_calls"] == 1
    assert recorded["agent_reset_calls"] == 1
    assert recorded["runtime_runs"] == 0
    assert recorded["stop_calls"] == 1
    assert recorded["environment_close_calls"] == 1
    assert "manipulate_prompts" not in recorded


def test_main_navigation_only_uses_dry_run_without_runtime(monkeypatch):
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {}

    _install_live_main_fakes(monkeypatch, recorded, planner_run_routine_result=True)

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

    args = main_module.Args(
        use_llm_planner=True,
        navigation_only=True,
        prompt="move to the table",
        skip_server_checks=True,
    )
    monkeypatch.setattr(args.planner, "validate_service_config", lambda: None)

    main_module.main(args)

    assert recorded["navigate_call"] == ("move to table", None, True)


def test_main_navigation_only_runs_navigation_once(monkeypatch):
    from examples.piper_real import main as main_module
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"navigate_calls": []}

    _install_live_main_fakes(monkeypatch, recorded, planner_run_routine_result=True)

    class FakeTaskDecomposer:
        def __init__(self, _config):
            pass

        def decompose(self, _prompt):
            return [
                task_decomposer_mod.Subtask(type="navigate", prompt="move to kitchen"),
                task_decomposer_mod.Subtask(type="navigate", prompt="move to living room"),
                task_decomposer_mod.Subtask(type="manipulate", prompt="pick cup"),
            ]

    def fake_navigate(prompt, ros_operator, *, dry_run=False):
        recorded["navigate_calls"].append((prompt, ros_operator, dry_run))
        return navigation_tool_mod.NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=0,
        )

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(main_module, "_run_required_server_checks", lambda *args, **kwargs: True)
    monkeypatch.setattr(navigation_tool_mod, "navigate", fake_navigate)

    args = main_module.Args(
        use_llm_planner=True,
        navigation_only=True,
        prompt="move to kitchen and living room",
        skip_server_checks=True,
    )
    monkeypatch.setattr(args.planner, "validate_service_config", lambda: None)

    main_module.main(args)

    assert recorded["navigate_calls"] == [("move to kitchen", None, True)]
