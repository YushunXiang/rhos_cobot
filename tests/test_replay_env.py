# tests/test_replay_env.py
"""Tests for HDF5 replay environment."""

import cv2
import h5py
import numpy as np
import pytest


def _create_test_hdf5(path: str, num_steps: int = 10, *, compressed: bool = True) -> str:
    """Create a minimal HDF5 episode file for testing."""
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["compress"] = compressed
        f.attrs["fps"] = 25

        qpos = np.random.randn(num_steps, 14).astype(np.float64)
        f.create_dataset("/observations/qpos", data=qpos)
        f.create_dataset("/observations/qvel", data=np.zeros((num_steps, 14)))
        f.create_dataset("/observations/effort", data=np.zeros((num_steps, 14)))
        f.create_dataset("/action", data=np.random.randn(num_steps, 14).astype(np.float64))
        f.create_dataset("/base_action", data=np.zeros((num_steps, 2)))

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
