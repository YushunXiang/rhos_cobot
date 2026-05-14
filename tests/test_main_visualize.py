import sys
import types
from types import SimpleNamespace


def test_main_visualize_module_exposes_visualize_args():
    from examples.piper_real import main_visualize

    args = main_visualize.Args(visualize=True, visualize_port=8123)

    assert args.visualize is True
    assert args.visualize_port == 8123
    assert main_visualize.__file__.endswith("main_visualize.py")


def test_main_visualize_live_hybrid_opens_updates_and_closes_visualizer(monkeypatch):
    from examples.piper_real import main_visualize
    from examples.piper_real import navigation_tool as navigation_tool_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {"contexts": [], "updates": [], "closed": 0}

    fake_env_module = types.ModuleType("examples.piper_real.env")
    fake_logger_module = types.ModuleType("examples.piper_real.logger")
    fake_replay_planner_module = types.ModuleType(
        "examples.piper_real.replay_manipulation_planner"
    )
    fake_visualizer_module = types.ModuleType("examples.piper_real.replay_visualizer")

    class FakeEnvironment:
        camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        fps = 25.0
        num_steps = 10_000
        save_obs = False
        saver = None

        def __init__(
            self,
            reset_position,
            prompt,
            robot_base_topic="/odom_raw",
            robot_base_cmd_topic="/cmd_vel",
        ):
            recorded["environment"] = self
            recorded["environment_init"] = {
                "reset_position": reset_position,
                "prompt": prompt,
                "robot_base_topic": robot_base_topic,
                "robot_base_cmd_topic": robot_base_cmd_topic,
            }
            self.ros_operator = SimpleNamespace(name="ros")
            self.cursor = 0

        def reset(self):
            recorded["environment_reset"] = recorded.get("environment_reset", 0) + 1

        def get_cursor(self):
            return self.cursor

        def set_prompt(self, prompt):
            recorded.setdefault("prompts", []).append(prompt)

        def refresh_observation_cache(self):
            recorded["refreshes"] = recorded.get("refreshes", 0) + 1

        def close(self):
            recorded["environment_close"] = recorded.get("environment_close", 0) + 1

    class FakeManipulationPlanner:
        def __init__(self, environment, config, task_memory_runtime=None):
            recorded["manipulation_planner"] = (
                environment,
                config,
                task_memory_runtime,
            )

    class FakeVisualizer:
        def __init__(self, environment, *, enabled, port, save_path):
            recorded["visualizer_init"] = {
                "environment": environment,
                "enabled": enabled,
                "port": port,
                "save_path": save_path,
            }

        def set_subtask_context(
            self,
            subtask_idx,
            total_subtasks,
            subtask_type,
            subtask_prompt,
            *,
            extra_info="",
        ):
            recorded["contexts"].append(
                (subtask_idx, total_subtasks, subtask_type, subtask_prompt, extra_info)
            )

        def update(self, step, *, extra_info=""):
            recorded["updates"].append((step, extra_info))
            return True

        def close(self):
            recorded["closed"] += 1

    fake_env_module.PiperRealEnvironment = FakeEnvironment
    fake_logger_module.InputJointStateLogger = lambda: None
    fake_logger_module.OutputJointStateLogger = lambda: None
    fake_logger_module.stitch_camera_videos = lambda _save_dir: None
    fake_replay_planner_module.ReplayManipulationPromptPlanner = FakeManipulationPlanner
    fake_visualizer_module.ReplayVisualizer = FakeVisualizer

    monkeypatch.setitem(sys.modules, "examples.piper_real.env", fake_env_module)
    monkeypatch.setitem(sys.modules, "examples.piper_real.logger", fake_logger_module)
    monkeypatch.setitem(
        sys.modules,
        "examples.piper_real.replay_manipulation_planner",
        fake_replay_planner_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "examples.piper_real.replay_visualizer",
        fake_visualizer_module,
    )

    import examples.piper_real as piper_real_package

    monkeypatch.setattr(piper_real_package, "env", fake_env_module, raising=False)
    monkeypatch.setattr(piper_real_package, "logger", fake_logger_module, raising=False)
    monkeypatch.setattr(
        piper_real_package,
        "replay_manipulation_planner",
        fake_replay_planner_module,
        raising=False,
    )
    monkeypatch.setattr(
        piper_real_package,
        "replay_visualizer",
        fake_visualizer_module,
        raising=False,
    )

    class FakeTaskDecomposer:
        def __init__(self, _config):
            pass

        def decompose(self, _prompt):
            return [
                task_decomposer_mod.Subtask(type="navigate", prompt="move to table"),
                task_decomposer_mod.Subtask(type="manipulate", prompt="pick cup"),
            ]

    class FakePolicyAgent:
        policy_metadata = {"reset_pose": [0.0] * 14}

        def reset(self):
            recorded["agent_reset"] = recorded.get("agent_reset", 0) + 1

    def fake_navigate(prompt, ros_operator, *, dry_run=False, frame_tick_callback=None):
        recorded["navigate"] = (prompt, ros_operator, dry_run)
        assert frame_tick_callback is not None
        environment = recorded["environment"]
        environment.cursor = 4
        frame_tick_callback()
        return navigation_tool_mod.NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name="default_demo",
            executed_steps=1,
        )

    def fake_run_manipulation_subtask(
        *_args,
        subtask_prompt,
        visualizer=None,
        **_kwargs,
    ):
        recorded["manipulation"] = (subtask_prompt, visualizer is not None)
        return {"executed_steps": 3, "prompt_queries": 1, "completed": True}

    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(navigation_tool_mod, "navigate", fake_navigate)
    monkeypatch.setattr(
        main_visualize,
        "_create_policy_agent",
        lambda _args: FakePolicyAgent(),
    )
    monkeypatch.setattr(
        main_visualize,
        "_build_ordered_task_memory_runtime",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        main_visualize,
        "_run_manipulation_subtask",
        fake_run_manipulation_subtask,
    )
    monkeypatch.setattr(
        main_visualize,
        "_run_required_server_checks",
        lambda *args, **kwargs: True,
    )

    args = main_visualize.Args(
        use_llm_planner=True,
        prompt="move and pick",
        skip_server_checks=True,
        visualize=True,
        visualize_port=8123,
    )
    monkeypatch.setattr(args.planner, "validate_service_config", lambda: None)
    monkeypatch.setattr(args.planner, "validate_motion_limits", lambda: None)

    main_visualize.main(args)

    assert recorded["visualizer_init"]["enabled"] is True
    assert recorded["visualizer_init"]["port"] == 8123
    assert recorded["contexts"] == [
        (1, 2, "navigate", "move to table", ""),
        (2, 2, "manipulate", "pick cup", ""),
    ]
    assert recorded["updates"] == [(4, "navigate subtask 1/2")]
    assert recorded["navigate"][0] == "move to table"
    assert recorded["navigate"][2] is True
    assert recorded["prompts"] == ["pick cup"]
    assert recorded["refreshes"] == 1
    assert recorded["manipulation"] == ("pick cup", True)
    assert recorded["closed"] == 1
    assert recorded["environment_close"] == 1


def test_main_visualize_stationary_runtime_streams_visualizer(monkeypatch):
    from openpi_client.runtime import runtime as runtime_mod
    from examples.piper_real import main_visualize

    recorded: dict[str, object] = {"contexts": [], "updates": [], "closed": 0}

    fake_env_module = types.ModuleType("examples.piper_real.env")
    fake_logger_module = types.ModuleType("examples.piper_real.logger")
    fake_visualizer_module = types.ModuleType("examples.piper_real.replay_visualizer")

    class FakeEnvironment:
        camera_names = ("cam_high",)
        fps = 25.0
        num_steps = 10_000

        def __init__(
            self,
            reset_position,
            prompt,
            robot_base_topic="/odom_raw",
            robot_base_cmd_topic="/cmd_vel",
        ):
            recorded["environment_init"] = {
                "reset_position": reset_position,
                "prompt": prompt,
                "robot_base_topic": robot_base_topic,
                "robot_base_cmd_topic": robot_base_cmd_topic,
            }
            self.cursor = 3

        def get_cursor(self):
            return self.cursor

    class FakeVisualizer:
        def __init__(self, environment, *, enabled, port, save_path):
            recorded["visualizer_init"] = {
                "environment": environment,
                "enabled": enabled,
                "port": port,
                "save_path": save_path,
            }

        def set_subtask_context(
            self,
            subtask_idx,
            total_subtasks,
            subtask_type,
            subtask_prompt,
            *,
            extra_info="",
        ):
            recorded["contexts"].append(
                (subtask_idx, total_subtasks, subtask_type, subtask_prompt, extra_info)
            )

        def update(self, step, *, extra_info=""):
            recorded["updates"].append((step, extra_info))
            return True

        def close(self):
            recorded["closed"] += 1

    class FakeRuntime:
        def __init__(
            self,
            *,
            environment,
            agent,
            subscribers,
            max_hz,
            num_episodes,
            max_episode_steps,
        ):
            recorded["runtime_init"] = {
                "environment": environment,
                "agent": agent,
                "subscribers": subscribers,
                "max_hz": max_hz,
                "num_episodes": num_episodes,
                "max_episode_steps": max_episode_steps,
            }

        def run(self):
            for subscriber in recorded["runtime_init"]["subscribers"]:
                subscriber.on_episode_start()
                subscriber.on_step({}, {"progress": 0.5})
                subscriber.on_episode_end()

    class FakePolicy:
        def get_server_metadata(self):
            return {"reset_pose": [0.0] * 14}

    fake_env_module.PiperRealEnvironment = FakeEnvironment
    fake_logger_module.InputJointStateLogger = lambda: None
    fake_logger_module.OutputJointStateLogger = lambda: None
    fake_visualizer_module.ReplayVisualizer = FakeVisualizer

    monkeypatch.setitem(sys.modules, "examples.piper_real.env", fake_env_module)
    monkeypatch.setitem(sys.modules, "examples.piper_real.logger", fake_logger_module)
    monkeypatch.setitem(
        sys.modules,
        "examples.piper_real.replay_visualizer",
        fake_visualizer_module,
    )

    import examples.piper_real as piper_real_package

    monkeypatch.setattr(piper_real_package, "env", fake_env_module, raising=False)
    monkeypatch.setattr(piper_real_package, "logger", fake_logger_module, raising=False)
    monkeypatch.setattr(
        piper_real_package,
        "replay_visualizer",
        fake_visualizer_module,
        raising=False,
    )
    monkeypatch.setattr(runtime_mod, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        main_visualize,
        "_create_remote_policy",
        lambda _args: FakePolicy(),
    )
    monkeypatch.setattr(
        main_visualize,
        "_run_required_server_checks",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        main_visualize._policy_agent,
        "PolicyAgent",
        lambda policy: ("agent", policy),
    )
    monkeypatch.setattr(
        main_visualize.action_chunk_broker,
        "ActionChunkBroker",
        lambda policy, action_horizon: ("broker", policy, action_horizon),
    )

    main_visualize.main(
        main_visualize.Args(
            prompt="pick cup",
            visualize=True,
            visualize_port=8123,
            skip_server_checks=True,
        )
    )

    assert recorded["visualizer_init"]["enabled"] is True
    assert recorded["contexts"] == [
        (1, 1, "policy", "pick cup", ""),
        (1, 1, "policy", "pick cup", ""),
    ]
    assert recorded["updates"] == [(2, "policy prompt: pick cup | progress: 0.5000")]
    assert recorded["closed"] == 1
