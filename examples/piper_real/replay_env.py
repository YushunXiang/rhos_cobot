"""HDF5 replay environment for offline inference debugging.

Implements the openpi_client Environment interface by reading pre-recorded
observations from an HDF5 episode file instead of querying live hardware.
Actions predicted by the policy are collected for comparison with the
ground-truth actions stored in the dataset.
"""

import logging
import math
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
        max_steps: int | None = None,
    ) -> None:
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._cursor: int = 0
        self._last_observation_idx: int | None = None
        self._dataset_path = dataset_path

        # Keep the HDF5 file open and decode frames lazily. Long-horizon
        # datasets can otherwise consume multiple gigabytes during startup.
        self._dataset = h5py.File(dataset_path, "r")
        self._compressed = bool(self._dataset.attrs.get("compress", False))
        self._fps = float(self._dataset.attrs.get("fps", 25.0))
        qpos = self._dataset["/observations/qpos"][()]
        if max_steps is not None and max_steps > 0:
            qpos = qpos[:max_steps]
        self._qpos = qpos
        self._num_steps: int = self._qpos.shape[0]

        self.ground_truth_actions = self._dataset["/action"][()][: self._num_steps]
        base_actions = self._dataset["/base_action"][()] if "base_action" in self._dataset else None
        self.ground_truth_base_actions = (
            base_actions[: self._num_steps] if base_actions is not None else None
        )
        self._recorded_base_actions: Optional[np.ndarray] = self.ground_truth_base_actions
        if self._recorded_base_actions is None and self.ground_truth_actions.ndim == 2:
            if self.ground_truth_actions.shape[1] >= 16:
                self._recorded_base_actions = self.ground_truth_actions[:, 14:16]
        self._image_datasets = {
            cam_name: self._dataset[f"/observations/images/{cam_name}"]
            for cam_name in self._dataset["/observations/images"].keys()
            if "_depth" not in cam_name
        }
        self._front_camera_name = self._select_front_camera_name()
        self._estimated_odometry = self._build_estimated_odometry()

        self.predicted_actions: list[np.ndarray] = []
        self.predicted_action_steps: list[int] = []

        logging.info(
            "ReplayEnvironment loaded %s (%d steps, cameras: %s, compressed=%s)",
            dataset_path,
            self._num_steps,
            list(self._image_datasets.keys()),
            self._compressed,
        )

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def front_camera_name(self) -> str:
        return self._front_camera_name

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._image_datasets.keys())

    @property
    def recorded_base_actions(self) -> Optional[np.ndarray]:
        return self._recorded_base_actions

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def get_cursor(self) -> int:
        return self._cursor

    def set_cursor(self, step_idx: int) -> None:
        if step_idx < 0 or step_idx > self._num_steps:
            raise IndexError(f"Replay cursor {step_idx} is out of range for {self._num_steps} steps")
        self._cursor = step_idx
        self._last_observation_idx = None

    def close(self) -> None:
        if getattr(self, "_dataset", None) is not None:
            self._dataset.close()
            self._dataset = None

    def __del__(self) -> None:
        self.close()

    def _load_image(self, cam_name: str, idx: int) -> np.ndarray:
        frame_data = self._image_datasets[cam_name][idx]
        if not self._compressed:
            return frame_data

        image = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"Failed to decode compressed frame {idx} from {cam_name} in {self._dataset_path}"
            )
        return image

    def _select_front_camera_name(self) -> str:
        if not self._image_datasets:
            raise RuntimeError(f"Replay dataset has no color cameras: {self._dataset_path}")

        preferred = ("cam_high", "cam_front", "front", "camera_front")
        for cam_name in preferred:
            if cam_name in self._image_datasets:
                return cam_name
        return next(iter(self._image_datasets))

    def _build_estimated_odometry(self) -> np.ndarray:
        odom = np.zeros((self._num_steps, 3), dtype=np.float64)
        if self._num_steps == 0 or self._recorded_base_actions is None:
            return odom

        dt = 1.0 / self._fps if self._fps > 0 else 0.0
        x = 0.0
        y = 0.0
        yaw = 0.0
        for step_idx in range(1, self._num_steps):
            linear_x, angular_z = self._recorded_base_actions[step_idx - 1]
            x += math.cos(yaw) * float(linear_x) * dt
            y += math.sin(yaw) * float(linear_x) * dt
            yaw += float(angular_z) * dt
            odom[step_idx] = (x, y, math.atan2(math.sin(yaw), math.cos(yaw)))
        return odom

    def get_image(self, cam_name: str, idx: int) -> np.ndarray:
        if cam_name not in self._image_datasets:
            raise KeyError(f"Unknown replay camera {cam_name!r}")
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"Replay step {idx} is out of range for {self._num_steps} steps")
        return self._load_image(cam_name, idx).copy()

    def get_state(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"Replay step {idx} is out of range for state")
        return self._qpos[idx].copy()

    def get_ground_truth_action(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.ground_truth_actions):
            raise IndexError(f"Replay step {idx} is out of range for actions")
        return np.asarray(self.ground_truth_actions[idx]).copy()

    def get_front_image(self, idx: int) -> np.ndarray:
        return self.get_image(self._front_camera_name, idx)

    def get_recorded_base_action(self, idx: int) -> Optional[np.ndarray]:
        if self._recorded_base_actions is None:
            return None
        if idx < 0 or idx >= len(self._recorded_base_actions):
            raise IndexError(f"Replay step {idx} is out of range for base actions")
        return np.asarray(self._recorded_base_actions[idx]).copy()

    def get_estimated_odometry(self, idx: int) -> dict[str, float]:
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"Replay step {idx} is out of range for odometry")
        x, y, yaw = self._estimated_odometry[idx]
        return {"x": float(x), "y": float(y), "yaw": float(yaw)}

    # -- Environment interface ------------------------------------------------

    @override
    def reset(self) -> None:
        self._cursor = 0
        self._last_observation_idx = None
        self.predicted_actions = []
        self.predicted_action_steps = []

    @override
    def is_episode_complete(self) -> bool:
        return self._cursor >= self._num_steps

    @override
    def get_observation(self) -> dict:
        idx = self._cursor
        if idx >= self._num_steps:
            raise IndexError(f"Replay exhausted at step {idx}/{self._num_steps}")

        images: dict[str, np.ndarray] = {}
        for cam_name in self._image_datasets:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    self._load_image(cam_name, idx),
                    self._render_height,
                    self._render_width,
                )
            )
            images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        self._last_observation_idx = idx
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
            step_idx = self._last_observation_idx
            if step_idx is None:
                step_idx = max(self._cursor - 1, 0)
            self.predicted_action_steps.append(step_idx)
            self._last_observation_idx = None
            logging.info(
                "Replay step %d — predicted action: %s%s",
                step_idx,
                np.array2string(
                    np.asarray(action["actions"]),
                    precision=4,
                    suppress_small=True,
                    max_line_width=200,
                ),
                (
                    ""
                    if "progress" not in action
                    else f", progress={float(np.asarray(action['progress'])):.4f}"
                ),
            )
