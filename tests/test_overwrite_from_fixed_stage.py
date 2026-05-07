import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.post_collect.overwrite_from_fixed_stage import (
    OverwritePlan,
    backup_path_for,
    inspect_fixed_file,
    overwrite_one,
    write_corrected_copy,
)


class OverwriteFromFixedStageTests(unittest.TestCase):
    def _make_episode(
        self,
        path: Path,
        n_frames: int,
        stage_values=None,
        fill_value: float = 0.0,
    ) -> None:
        with h5py.File(path, "w") as root:
            root.attrs["fps"] = 25
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=np.full((n_frames, 14), fill_value, dtype=np.float32))
            images = obs.create_group("images")
            images.create_dataset(
                "cam_high",
                data=np.array([f"img_{idx}".encode("utf-8") for idx in range(n_frames)], dtype="S16"),
            )
            root.create_dataset("action", data=np.full((n_frames, 14), fill_value + 1, dtype=np.float32))
            root.create_dataset("base_action", data=np.full((n_frames, 2), fill_value + 2, dtype=np.float32))
            root.create_dataset("meta_scalar", data=np.array(7, dtype=np.int64))
            if stage_values is not None:
                root.create_dataset("stage", data=np.asarray(stage_values, dtype=np.int64))

    def test_inspect_fixed_file_counts_stage_three_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_0.hdf5"
            self._make_episode(path, n_frames=5, stage_values=[0, 1, 2, 3, 3])
            n_frames, drop_count = inspect_fixed_file(path)
            self.assertEqual(n_frames, 5)
            self.assertEqual(drop_count, 2)

    def test_write_corrected_copy_removes_stage_three_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixed_path = Path(tmp_dir) / "episode_0.hdf5"
            target_path = Path(tmp_dir) / "episode_0.hdf5.tmp"
            self._make_episode(fixed_path, n_frames=5, stage_values=[0, 1, 2, 3, 3], fill_value=5.0)

            write_corrected_copy(fixed_path, target_path, drop_count=2)

            with h5py.File(target_path, "r") as root:
                self.assertEqual(root["/observations/qpos"].shape[0], 3)
                self.assertEqual(root["/observations/images/cam_high"].shape[0], 3)
                self.assertEqual(root["/action"].shape[0], 3)
                self.assertEqual(root["/base_action"].shape[0], 3)
                self.assertEqual(root["/stage"][()].tolist(), [0, 1, 2])
                self.assertEqual(int(root["/meta_scalar"][()]), 7)

    def test_overwrite_one_renames_source_to_bkup_and_removes_stage_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / "episode_1.hdf5"
            fixed_path = root / "fixed_episode_1.hdf5"
            self._make_episode(source_path, n_frames=4, stage_values=[0, 0, 0, 0], fill_value=9.0)
            self._make_episode(fixed_path, n_frames=5, stage_values=[0, 1, 2, 3, 3], fill_value=2.0)

            plan = OverwritePlan(
                name="episode_1.hdf5",
                fixed_path=fixed_path,
                source_path=source_path,
                backup_path=backup_path_for(source_path),
                size_bytes=fixed_path.stat().st_size,
                fixed_frames=5,
                drop_count=2,
            )

            overwrite_one(plan)

            self.assertTrue(plan.backup_path.exists())
            with h5py.File(plan.backup_path, "r") as backup_root:
                self.assertEqual(backup_root["/stage"][()].tolist(), [0, 0, 0, 0])
                self.assertTrue(np.allclose(backup_root["/observations/qpos"][()], 9.0))

            with h5py.File(source_path, "r") as source_root:
                self.assertEqual(source_root["/stage"][()].tolist(), [0, 1, 2])
                self.assertEqual(source_root["/observations/qpos"].shape[0], 3)
                self.assertTrue(np.allclose(source_root["/observations/qpos"][()], 2.0))


if __name__ == "__main__":
    unittest.main()
