import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.post_collect.review_stage_web import (
    ReviewError,
    build_segments,
    normalize_stage_payload,
    write_stage_array,
)


class ReviewStageWebTests(unittest.TestCase):
    def _make_episode(self, root_dir: Path, name: str, n_frames: int = 4, with_stage: bool = False) -> Path:
        path = root_dir / name
        with h5py.File(path, "w") as root:
            root.attrs["fps"] = 25
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=np.zeros((n_frames, 14), dtype=np.float32))
            images = obs.create_group("images")
            images.create_dataset("cam_high", data=np.array([b"img"] * n_frames, dtype="S8"))
            root.create_dataset("action", data=np.zeros((n_frames, 14), dtype=np.float32))
            root.create_dataset("base_action", data=np.zeros((n_frames, 2), dtype=np.float32))
            if with_stage:
                root.create_dataset("stage", data=np.zeros((n_frames,), dtype=np.int64))
        return path

    def test_build_segments(self) -> None:
        segments = build_segments([0, 0, 1, 1, 1, 2])
        self.assertEqual(
            segments,
            [
                {"start": 0, "end": 2, "stage": 0},
                {"start": 2, "end": 5, "stage": 1},
                {"start": 5, "end": 6, "stage": 2},
            ],
        )

    def test_normalize_stage_payload_accepts_segments(self) -> None:
        stage = normalize_stage_payload(
            {
                "segments": [
                    {"start": 0, "end": 2, "stage": 0},
                    {"start": 2, "end": 5, "stage": 2},
                ]
            },
            n_frames=5,
        )
        self.assertEqual(stage, [0, 0, 2, 2, 2])

    def test_normalize_stage_payload_rejects_gaps(self) -> None:
        with self.assertRaises(ReviewError):
            normalize_stage_payload(
                {
                    "segments": [
                        {"start": 0, "end": 2, "stage": 0},
                        {"start": 3, "end": 5, "stage": 1},
                    ]
                },
                n_frames=5,
            )

    def test_write_stage_array_creates_fixed_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            source_path = self._make_episode(root_dir, "episode_0.hdf5", n_frames=4, with_stage=False)

            target_path = write_stage_array(
                source_path=source_path,
                stage_values=[0, 1, 1, 2],
                save_mode="fixed",
                fixed_dir_name="fixed_stage",
            )

            self.assertEqual(target_path, root_dir / "fixed_stage" / "episode_0.hdf5")
            self.assertTrue(target_path.exists())
            with h5py.File(source_path, "r") as source_file:
                self.assertNotIn("stage", source_file)
            with h5py.File(target_path, "r") as fixed_file:
                self.assertEqual(fixed_file["/stage"][()].tolist(), [0, 1, 1, 2])

    def test_write_stage_array_overwrites_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            source_path = self._make_episode(root_dir, "episode_1.hdf5", n_frames=3, with_stage=True)

            target_path = write_stage_array(
                source_path=source_path,
                stage_values=[0, 2, 2],
                save_mode="overwrite",
                fixed_dir_name="fixed_stage",
            )

            self.assertEqual(target_path, source_path)
            with h5py.File(source_path, "r") as source_file:
                self.assertEqual(source_file["/stage"][()].tolist(), [0, 2, 2])


if __name__ == "__main__":
    unittest.main()
