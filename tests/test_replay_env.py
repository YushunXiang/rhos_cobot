# tests/test_replay_env.py
"""Tests for HDF5 replay environment."""

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

    def test_ground_truth_actions_accessible(self, tmp_path):
        hdf5_path = str(tmp_path / "episode_0.hdf5")
        _create_test_hdf5(hdf5_path, num_steps=5)
        from examples.piper_real.replay_env import ReplayEnvironment
        env = ReplayEnvironment(dataset_path=hdf5_path, prompt="test task")
        assert env.ground_truth_actions.shape == (5, 14)

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
