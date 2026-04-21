# tests/test_replay_env.py
"""Tests for HDF5 replay environment."""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest


def _create_test_hdf5(
    path: str,
    num_steps: int = 10,
    *,
    compressed: bool = True,
    fps: float = 25.0,
    action_dim: int = 14,
    include_base_action: bool = True,
    base_actions: np.ndarray | None = None,
) -> str:
    """Create a minimal HDF5 episode file for testing."""
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["compress"] = compressed
        f.attrs["fps"] = fps

        qpos = np.random.randn(num_steps, 14).astype(np.float64)
        f.create_dataset("/observations/qpos", data=qpos)
        f.create_dataset("/observations/qvel", data=np.zeros((num_steps, 14)))
        f.create_dataset("/observations/effort", data=np.zeros((num_steps, 14)))
        f.create_dataset("/action", data=np.random.randn(num_steps, action_dim).astype(np.float64))
        if include_base_action:
            if base_actions is None:
                base_actions = np.zeros((num_steps, 2), dtype=np.float64)
            f.create_dataset("/base_action", data=np.asarray(base_actions, dtype=np.float64))

        for cam_name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if compressed:
                dt = h5py.special_dtype(vlen=np.uint8)
                ds = f.create_dataset(
                    f"/observations/images/{cam_name}",
                    shape=(num_steps,),
                    dtype=dt,
                )
                for i in range(num_steps):
                    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    _, buf = cv2.imencode(".jpg", img)
                    ds[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
            else:
                imgs = np.random.randint(0, 255, (num_steps, 480, 640, 3), dtype=np.uint8)
                f.create_dataset(f"/observations/images/{cam_name}", data=imgs)
    return path


class TestReplayEnvironment:
    def test_init_loads_episode(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        assert env._num_steps == 5

    def test_reset_sets_cursor_to_zero(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        assert env._cursor == 0

    def test_get_observation_returns_correct_format(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        obs = env.get_observation()
        assert "state" in obs
        assert "images" in obs
        assert "prompt" in obs
        assert obs["state"].shape == (14,)
        assert obs["prompt"] == "test task"
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert cam in obs["images"]
            assert obs["images"][cam].shape == (3, 224, 224)
            assert obs["images"][cam].dtype == np.uint8

    def test_get_observation_advances_cursor(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.get_observation()
        assert env._cursor == 1

    def test_set_prompt_updates_future_observations(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.set_prompt("updated task")

        obs = env.get_observation()

        assert obs["prompt"] == "updated task"

    def test_set_cursor_repositions_replay(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.set_cursor(3)

        assert env.get_cursor() == 3
        obs = env.get_observation()
        assert np.allclose(obs["state"], env._qpos[3])

    def test_is_episode_complete_at_end(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=2)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        assert not env.is_episode_complete()
        env.get_observation()
        assert not env.is_episode_complete()
        env.get_observation()
        assert env.is_episode_complete()

    def test_apply_action_logs_action(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=3)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.get_observation()
        fake_action = {"actions": np.zeros(14)}
        env.apply_action(fake_action)
        assert len(env.predicted_actions) == 1

    def test_apply_action_tracks_observation_step(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        env.set_cursor(2)
        env.get_observation()
        env.apply_action({"actions": np.zeros(14)})

        assert env.predicted_action_steps == [2]

    def test_ground_truth_actions_accessible(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        assert env.ground_truth_actions.shape == (5, 14)

    def test_get_state_returns_copy(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        state = env.get_state(0)
        assert state.shape == (14,)
        state[:] = 123
        assert not np.allclose(state, env.get_state(0))

    def test_get_ground_truth_action_returns_copy(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        action = env.get_ground_truth_action(0)
        assert action.shape == (14,)
        action[:] = 321
        assert not np.allclose(action, env.get_ground_truth_action(0))

    def test_uncompressed_images(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=3, compressed=False)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        env.reset()
        obs = env.get_observation()
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert obs["images"][cam].shape == (3, 224, 224)
            assert obs["images"][cam].dtype == np.uint8

    def test_front_camera_defaults_to_cam_high(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=3)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")

        assert env.front_camera_name == "cam_high"

    def test_estimated_odometry_integrates_base_action(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        base_actions = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        _create_test_hdf5(hdf5_path, num_steps=3, fps=2.0, base_actions=base_actions)
        from examples.piper_real.replay_env import ReplayEnvironment

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")

        odom0 = env.get_estimated_odometry(0)
        odom1 = env.get_estimated_odometry(1)
        odom2 = env.get_estimated_odometry(2)

        assert odom0 == {"x": 0.0, "y": 0.0, "yaw": 0.0}
        assert np.isclose(odom1["x"], 0.5)
        assert np.isclose(odom1["y"], 0.0)
        assert np.isclose(odom2["x"], 1.0)
        assert np.isclose(odom2["y"], 0.0)


class TestOfflineReplayNavigationPlanner:
    def test_run_advances_by_duration_times_fps(self, tmp_path, monkeypatch):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5, fps=2.0)

        from examples.piper_real.planner_config import PlannerConfig
        from examples.piper_real.replay_env import ReplayEnvironment
        from examples.piper_real.replay_planner import OfflineReplayNavigationPlanner

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        planner = OfflineReplayNavigationPlanner(env, PlannerConfig(base_url="http://unused", model="test"))
        responses = iter(
            [
                (
                    '{"action":"move","linear_x":0.2,"angular_z":0.0,"duration":1.0,"reasoning":"forward"}',
                    {
                        "action": "move",
                        "linear_x": 0.2,
                        "angular_z": 0.0,
                        "duration": 1.0,
                        "reasoning": "forward",
                    },
                ),
                (
                    '{"action":"stop","reason":"done"}',
                    {
                        "action": "stop",
                        "reason": "done",
                    },
                ),
            ]
        )
        monkeypatch.setattr(planner, "query_llm", lambda *_args, **_kwargs: next(responses))

        assert planner.run("move to table") is True
        assert planner.current_step == 2
        assert any(cmd == [0.2, 0.0] for cmd in planner.replay_commands)
        assert planner._history[0]["command"]["replay_frames_advanced"] == 2

    def test_run_fails_when_replay_is_exhausted(self, tmp_path, monkeypatch):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=2, fps=1.0)

        from examples.piper_real.planner_config import PlannerConfig
        from examples.piper_real.replay_env import ReplayEnvironment
        from examples.piper_real.replay_planner import OfflineReplayNavigationPlanner

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        planner = OfflineReplayNavigationPlanner(env, PlannerConfig(base_url="http://unused", model="test"))
        responses = iter(
            [
                (
                    '{"action":"move","linear_x":0.2,"angular_z":0.0,"duration":1.0,"reasoning":"forward"}',
                    {
                        "action": "move",
                        "linear_x": 0.2,
                        "angular_z": 0.0,
                        "duration": 1.0,
                        "reasoning": "forward",
                    },
                ),
                (
                    '{"action":"move","linear_x":0.2,"angular_z":0.0,"duration":1.0,"reasoning":"forward"}',
                    {
                        "action": "move",
                        "linear_x": 0.2,
                        "angular_z": 0.0,
                        "duration": 1.0,
                        "reasoning": "forward",
                    },
                ),
            ]
        )
        monkeypatch.setattr(planner, "query_llm", lambda *_args, **_kwargs: next(responses))

        assert planner.run("move until replay ends") is False
        assert planner.current_step == 1
        assert env.get_cursor() == env.num_steps

    def test_planner_syncs_to_environment_cursor_before_run(self, tmp_path, monkeypatch):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5, fps=2.0)

        from examples.piper_real.planner_config import PlannerConfig
        from examples.piper_real.replay_env import ReplayEnvironment
        from examples.piper_real.replay_planner import OfflineReplayNavigationPlanner

        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        planner = OfflineReplayNavigationPlanner(env, PlannerConfig(base_url="http://unused", model="test"))
        env.set_cursor(3)
        responses = iter(
            [
                (
                    '{"action":"stop","reason":"aligned"}',
                    {
                        "action": "stop",
                        "reason": "aligned",
                    },
                ),
            ]
        )
        monkeypatch.setattr(planner, "query_llm", lambda *_args, **_kwargs: next(responses))

        assert planner.run("resume from current cursor") is True
        assert planner.current_step == 3
        assert env.get_cursor() == 3


class TestMainReplayIntegration:
    """Verify that main.py accepts --replay-dataset and constructs ReplayEnvironment."""

    def test_args_has_replay_dataset_field(self):
        import dataclasses
        from examples.piper_real.main import Args

        fields = {f.name for f in dataclasses.fields(Args)}
        assert "replay_dataset" in fields

    def test_args_replay_dataset_default_is_empty(self):
        from examples.piper_real.main import Args

        args = Args()
        assert args.replay_dataset == ""

    def test_main_routes_replay_llm_planner_to_new_branch(self, monkeypatch):
        from examples.piper_real import main as main_module

        observed: dict[str, str] = {}

        def fake_replay_planner(args, prompt):
            observed["dataset"] = args.replay_dataset
            observed["prompt"] = prompt

        monkeypatch.setattr(main_module, "_run_replay_planner", fake_replay_planner)

        main_module.main(
            main_module.Args(
                replay_dataset="/tmp/fake_episode.hdf5",
                use_llm_planner=True,
                prompt="navigate to the table",
            )
        )

        assert observed == {
            "dataset": "/tmp/fake_episode.hdf5",
            "prompt": "navigate to the table",
        }

    def test_main_routes_replay_hybrid_to_new_branch(self, monkeypatch):
        from examples.piper_real import main as main_module

        observed: dict[str, str] = {}

        def fake_replay_hybrid(args, prompt):
            observed["dataset"] = args.replay_dataset
            observed["prompt"] = prompt

        monkeypatch.setattr(main_module, "_run_replay_hybrid", fake_replay_hybrid)

        main_module.main(
            main_module.Args(
                replay_dataset="/tmp/fake_episode.hdf5",
                replay_mode="hybrid",
                prompt="pick up the red cup",
            )
        )

        assert observed == {
            "dataset": "/tmp/fake_episode.hdf5",
            "prompt": "pick up the red cup",
        }

    def test_run_replay_hybrid_shares_cursor_between_subtasks(self, monkeypatch):
        from examples.piper_real import main as main_module
        from examples.piper_real import replay_env as replay_env_mod
        from examples.piper_real import replay_manipulation_planner as replay_manipulation_planner_mod
        from examples.piper_real import task_decomposer as task_decomposer_mod
        from examples.piper_real.planner_config import PlannerConfig

        recorded: dict[str, object] = {}

        class FakeReplayEnvironment:
            def __init__(self, dataset_path: str, prompt: str, max_steps: int | None = None) -> None:
                recorded["dataset_path"] = dataset_path
                recorded["prompt"] = prompt
                recorded["max_steps"] = max_steps
                recorded["closed"] = False
                self._cursor = 0
                self.num_steps = 10
                self.fps = 25.0
                self.front_camera_name = "cam_high"
                self.camera_names = ("cam_high",)
                self.predicted_actions: list[np.ndarray] = []
                self.predicted_action_steps: list[int] = []
                self.ground_truth_actions = np.zeros((self.num_steps, 14), dtype=np.float32)
                self.ground_truth_base_actions = None

            def reset(self) -> None:
                self._cursor = 0
                self.predicted_actions = []
                self.predicted_action_steps = []

            def close(self) -> None:
                recorded["closed"] = True

            def set_prompt(self, prompt: str) -> None:
                recorded.setdefault("manipulate_prompts", []).append(prompt)

            def get_cursor(self) -> int:
                return self._cursor

            def set_cursor(self, step_idx: int) -> None:
                self._cursor = step_idx

            def is_episode_complete(self) -> bool:
                return self._cursor >= self.num_steps

            def get_observation(self) -> dict:
                idx = self._cursor
                if idx >= self.num_steps:
                    raise IndexError("Replay exhausted")
                self._cursor += 1
                return {
                    "state": np.zeros(14, dtype=np.float32),
                    "images": {},
                    "prompt": "unused",
                    "step": idx,
                }

            def apply_action(self, action: dict) -> None:
                self.predicted_actions.append(np.asarray(action["actions"]))
                self.predicted_action_steps.append(self._cursor - 1)

        class FakeTaskDecomposer:
            def __init__(self, _config) -> None:
                pass

            def decompose(self, _prompt: str):
                return [
                    task_decomposer_mod.Subtask(type="navigate", prompt="nav to sink"),
                    task_decomposer_mod.Subtask(type="manipulate", prompt="pick plate"),
                    task_decomposer_mod.Subtask(type="navigate", prompt="nav to table"),
                ]

        class FakePolicyAgent:
            def reset(self) -> None:
                recorded["policy_resets"] = int(recorded.get("policy_resets", 0)) + 1

            def get_action(self, observation: dict) -> dict:
                recorded.setdefault("policy_obs_steps", []).append(observation["step"])
                return {"actions": np.zeros(14, dtype=np.float32)}

        class FakeManipulationPromptPlanner:
            def __init__(self, _environment, _config) -> None:
                pass

            def plan(
                self,
                *,
                task_prompt: str,
                current_policy_prompt: str,
                executed_policy_steps: int,
                prompt_history,
            ):
                recorded.setdefault("replan_calls", []).append(
                    {
                        "task_prompt": task_prompt,
                        "current_policy_prompt": current_policy_prompt,
                        "executed_policy_steps": executed_policy_steps,
                        "history_len": len(prompt_history),
                    }
                )
                if executed_policy_steps == 0:
                    return replay_manipulation_planner_mod.ManipulationReplanDecision(
                        action="continue",
                        prompt="grasp the plate rim",
                        reason="stage aligned",
                    )
                return replay_manipulation_planner_mod.ManipulationReplanDecision(
                    action="complete",
                    reason="plate is secured",
                )

        monkeypatch.setattr(replay_env_mod, "ReplayEnvironment", FakeReplayEnvironment)
        monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
        monkeypatch.setattr(
            replay_manipulation_planner_mod,
            "ReplayManipulationPromptPlanner",
            FakeManipulationPromptPlanner,
        )
        monkeypatch.setattr(main_module, "_create_policy_agent", lambda _args: FakePolicyAgent())

        args = main_module.Args(
            replay_dataset="/tmp/episode_4.hdf5",
            replay_mode="hybrid",
            replay_manipulate_max_steps=4,
            replay_manipulate_replan_interval_steps=2,
            skip_server_checks=True,
            prompt="long-horizon replay mock validation",
            planner=PlannerConfig(base_url="http://unused", model="test"),
        )

        main_module._run_replay_hybrid(args, args.prompt)

        # navigate subtasks are skipped in replay mode: no cursor advance, no tool run.
        assert "navigate_starts" not in recorded
        assert recorded["manipulate_prompts"] == ["grasp the plate rim"]
        # Manipulate runs directly from cursor 0 since navigate subtasks no longer advance it.
        assert recorded["policy_obs_steps"] == [0, 1]
        assert [call["executed_policy_steps"] for call in recorded["replan_calls"]] == [0, 2]
        assert recorded["closed"] is True

    def test_run_manipulation_subtask_exports_debug_on_cap(self, tmp_path):
        from examples.piper_real import main as main_module
        from examples.piper_real import replay_manipulation_planner as replay_manipulation_planner_mod

        class FakeReplayEnvironment:
            def __init__(self) -> None:
                self._cursor = 0
                self.num_steps = 8
                self.camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")

            def get_cursor(self) -> int:
                return self._cursor

            def set_prompt(self, _prompt: str) -> None:
                pass

            def is_episode_complete(self) -> bool:
                return self._cursor >= self.num_steps

            def get_observation(self) -> dict:
                idx = self._cursor
                self._cursor += 1
                return {
                    "state": np.zeros(14, dtype=np.float32),
                    "images": {},
                    "prompt": "unused",
                    "step": idx,
                }

            def apply_action(self, _action: dict) -> None:
                pass

            def get_image(self, cam_name: str, idx: int) -> np.ndarray:
                color_seed = {
                    "cam_high": (40, 80, 120),
                    "cam_left_wrist": (80, 120, 160),
                    "cam_right_wrist": (120, 160, 200),
                }[cam_name]
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:] = tuple(min(channel + idx * 3, 255) for channel in color_seed)
                return frame

        class FakePolicyAgent:
            policy_metadata: dict[str, object] = {}

            def reset(self) -> None:
                pass

            def get_action(self, _observation: dict) -> dict:
                return {"actions": np.zeros(14, dtype=np.float32)}

        class FakeManipulationPromptPlanner:
            def plan(
                self,
                *,
                task_prompt: str,
                current_policy_prompt: str,
                executed_policy_steps: int,
                prompt_history,
            ):
                if executed_policy_steps == 0:
                    return replay_manipulation_planner_mod.ManipulationReplanDecision(
                        action="continue",
                        prompt=f"{task_prompt} / stage 0",
                        reason="initial alignment",
                    )
                return replay_manipulation_planner_mod.ManipulationReplanDecision(
                    action="continue",
                    prompt=f"{current_policy_prompt} / step {executed_policy_steps}",
                    reason=f"need more progress at {executed_policy_steps}",
                )

        env = FakeReplayEnvironment()
        result = main_module._run_manipulation_subtask(
            env,
            FakePolicyAgent(),
            FakeManipulationPromptPlanner(),
            subtask_prompt="Position the plate over the sink.",
            max_steps=4,
            replan_interval_steps=2,
            progress_complete_threshold=0.95,
            progress_stall_threshold=0.01,
            progress_stall_steps=3,
            progress_regression_threshold=0.05,
            progress_confirm_with_replanner=True,
            debug_export_dir=str(tmp_path),
            subtask_index=2,
            total_subtasks=9,
            visualizer=None,
        )

        assert result["executed_steps"] == 4

        export_dir = tmp_path / "subtask_02_position_the_plate_over_the_sink"
        manifest_path = export_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["subtask_index"] == 2
        assert manifest["max_steps"] == 4
        assert manifest["executed_steps"] == 4
        assert manifest["history_window_size"] == 2
        assert (export_dir / "final_frame.png").exists()
        for entry in manifest["recent_prompt_history"]:
            assert Path(entry["image_path"]).exists()

    def test_run_replay_hybrid_aborts_when_manipulate_subtask_hits_cap(self, monkeypatch):
        from examples.piper_real import main as main_module
        from examples.piper_real import replay_env as replay_env_mod
        from examples.piper_real import replay_visualizer as replay_visualizer_mod
        from examples.piper_real import task_decomposer as task_decomposer_mod
        from examples.piper_real.planner_config import PlannerConfig

        recorded: dict[str, object] = {}

        class FakeReplayEnvironment:
            def __init__(self, dataset_path: str, prompt: str, max_steps: int | None = None) -> None:
                recorded["dataset_path"] = dataset_path
                recorded["prompt"] = prompt
                self._cursor = 0
                self.num_steps = 20
                self.fps = 25.0
                self.front_camera_name = "cam_high"
                self.camera_names = ("cam_high",)

            def close(self) -> None:
                recorded["closed"] = True

            def get_cursor(self) -> int:
                return self._cursor

            def set_cursor(self, step_idx: int) -> None:
                self._cursor = step_idx

            def is_episode_complete(self) -> bool:
                return False

        class FakeTaskDecomposer:
            def __init__(self, _config) -> None:
                pass

            def decompose(self, _prompt: str):
                return [
                    task_decomposer_mod.Subtask(type="navigate", prompt="nav to sink"),
                    task_decomposer_mod.Subtask(type="manipulate", prompt="pick plate"),
                    task_decomposer_mod.Subtask(type="navigate", prompt="nav to table"),
                ]

        class FakeReplayVisualizer:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def set_subtask_context(self, *_args, **_kwargs) -> None:
                pass

            def update(self, *_args, **_kwargs) -> bool:
                return True

            def close(self) -> None:
                pass

        monkeypatch.setattr(replay_env_mod, "ReplayEnvironment", FakeReplayEnvironment)
        monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
        monkeypatch.setattr(replay_visualizer_mod, "ReplayVisualizer", FakeReplayVisualizer)
        monkeypatch.setattr(main_module, "_create_policy_agent", lambda _args: object())
        monkeypatch.setattr(
            main_module,
            "_run_manipulation_subtask",
            lambda *_args, **_kwargs: {
                "executed_steps": 200,
                "prompt_queries": 13,
                "completed": False,
                "completed_by_replan": False,
                "completed_by_progress": False,
                "last_policy_prompt": "Move the center plate to the faucet",
                "stop_reason": "step_cap",
            },
        )

        args = main_module.Args(
            replay_dataset="/tmp/episode_4.hdf5",
            replay_mode="hybrid",
            replay_manipulate_max_steps=200,
            replay_manipulate_replan_interval_steps=16,
            skip_server_checks=True,
            prompt="test hybrid prompt",
            planner=PlannerConfig(base_url="http://unused", model="test"),
        )

        main_module._run_replay_hybrid(args, args.prompt)

        # navigate subtasks are skipped in replay mode; the abort is driven by the manipulate cap.
        assert "navigate_prompts" not in recorded
        assert recorded["closed"] is True
