"""HDF5 replay environment for offline inference debugging.

Implements the openpi_client Environment interface by reading pre-recorded
observations from an HDF5 episode file instead of querying live hardware.
Actions predicted by the policy are collected for comparison with the
ground-truth actions stored in the dataset.
"""

import logging
from typing import Optional

import cv2
import einops
import h5py
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class ReplayEnvironment(_environment.Environment):
    """Replays observations from an HDF5 episode file.

    Parameters
    ----------
    dataset_path:
        Absolute path to a single ``episode_*.hdf5`` file.
    prompt:
        Task description forwarded to the policy.
    render_height, render_width:
        Target image resolution (must match policy training config).
    """

    def __init__(
        self,
        dataset_path: str,
        prompt: str = "",
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._cursor: int = 0

        # --- load dataset ---------------------------------------------------
        with h5py.File(dataset_path, "r") as f:
            compressed = f.attrs.get("compress", False)
            self._qpos: np.ndarray = f["/observations/qpos"][()]
            self._num_steps: int = self._qpos.shape[0]

            self.ground_truth_actions: np.ndarray = f["/action"][()]
            self.ground_truth_base_actions: Optional[np.ndarray] = (
                f["/base_action"][()] if "base_action" in f else None
            )

            # Decode images eagerly — dataset fits in memory for typical
            # episode lengths (< 2000 steps).
            self._images: dict[str, list[np.ndarray]] = {}
            for cam_name in f["/observations/images"].keys():
                if "_depth" in cam_name:
                    continue
                raw = f[f"/observations/images/{cam_name}"][()]
                frames: list[np.ndarray] = []
                for frame_data in raw:
                    if compressed:
                        img = cv2.imdecode(
                            np.frombuffer(frame_data, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                    else:
                        img = frame_data
                    frames.append(img)
                self._images[cam_name] = frames

        self.predicted_actions: list[np.ndarray] = []

        logging.info(
            "ReplayEnvironment loaded %s (%d steps, cameras: %s)",
            dataset_path,
            self._num_steps,
            list(self._images.keys()),
        )

    # -- Environment interface ------------------------------------------------

    @override
    def reset(self) -> None:
        self._cursor = 0
        self.predicted_actions = []

    @override
    def is_episode_complete(self) -> bool:
        return self._cursor >= self._num_steps

    @override
    def get_observation(self) -> dict:
        idx = self._cursor
        if idx >= self._num_steps:
            raise IndexError(f"Replay exhausted at step {idx}/{self._num_steps}")

        images: dict[str, np.ndarray] = {}
        for cam_name, frames in self._images.items():
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    frames[idx], self._render_height, self._render_width
                )
            )
            images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        self._cursor += 1

        return {
            "state": self._qpos[idx].copy(),
            "images": images,
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        if "actions" in action:
            self.predicted_actions.append(np.array(action["actions"]))
            step_idx = len(self.predicted_actions) - 1
            logging.info(
                "Replay step %d — predicted action[:4]: %s",
                step_idx,
                action["actions"][:4],
            )
