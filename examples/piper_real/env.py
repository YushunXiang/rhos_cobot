
import sys
from types import SimpleNamespace
from typing import List, Optional  # noqa: UP035

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    from examples.piper_real import real_env as _real_env
    from examples.piper_real.logger import ModelInputObservationSaver as _obs_saver
except ModuleNotFoundError:
    import real_env as _real_env
    from logger import ModelInputObservationSaver as _obs_saver


_DEFAULT_CAMERA_NAMES: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_DEFAULT_FRONT_CAMERA = "cam_high"
_DEFAULT_FPS = 25.0


class PiperRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware.

    Exposes a ReplayEnvironment-compatible read-only surface
    (``get_image``, ``get_cursor``, ``num_steps``, ``camera_names``, ``fps``,
    ``front_camera_name``) so that the manipulation replanner and ordered
    task-memory runtime can drive real-robot deployments with the same code
    path used for offline replay.
    """

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        prompt: str = "",
        robot_base_topic: str = _real_env.DEFAULT_ROBOT_BASE_TOPIC,
        robot_base_cmd_topic: str = _real_env.DEFAULT_ROBOT_BASE_CMD_TOPIC,
    ) -> None:
        self._env = _real_env.make_real_env(
            init_node=True,
            reset_position=reset_position,
            robot_base_topic=robot_base_topic,
            robot_base_cmd_topic=robot_base_cmd_topic,
        )
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None
        self._last_raw_images: dict[str, np.ndarray] = {}
        self._step_counter: int = 0
        self.save_obs = True
        self.frame_cnt = 0
        if self.save_obs:
            self.saver = _obs_saver()

    @property
    def ros_operator(self):
        return self._env.ros_operator

    @property
    def camera_names(self) -> tuple[str, ...]:
        if self._last_raw_images:
            return tuple(self._last_raw_images.keys())
        return _DEFAULT_CAMERA_NAMES

    @property
    def front_camera_name(self) -> str:
        return _DEFAULT_FRONT_CAMERA

    @property
    def num_steps(self) -> int:
        # Real robot has no fixed horizon; return a large value so replanner
        # bounds checks (``idx < num_steps``) always pass.
        return sys.maxsize

    @property
    def fps(self) -> float:
        return _DEFAULT_FPS

    def get_cursor(self) -> int:
        return self._step_counter

    def get_image(self, cam_name: str, idx: int) -> np.ndarray:
        # ``idx`` is accepted for replay-interface compatibility but ignored:
        # the real robot only exposes the most recent frame.
        del idx
        if not self._last_raw_images:
            raise RuntimeError(
                "No camera frame cached yet. Call reset()/get_observation() before get_image()."
            )
        if cam_name not in self._last_raw_images:
            raise KeyError(f"Unknown camera {cam_name!r}")
        return self._last_raw_images[cam_name].copy()

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _cache_raw_images_from_timestep(self) -> None:
        if self._ts is None:
            self._last_raw_images = {}
            return

        images = self._ts.observation.get("images", {})
        self._last_raw_images = {
            cam_name: np.ascontiguousarray(image)
            for cam_name, image in images.items()
            if "_depth" not in cam_name
        }

    def refresh_observation_cache(self) -> None:
        """Refresh cached camera frames without publishing a robot action."""
        self._ts = SimpleNamespace(observation=self._env.get_observation())
        self._cache_raw_images_from_timestep()

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()
        self._cache_raw_images_from_timestep()
        self._step_counter = 0

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        # Cache the raw HWC uint8 frame for VLM replanner / task memory before
        # we resize + rearrange for policy input.
        self._cache_raw_images_from_timestep()
        self._step_counter += 1

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        if self.save_obs:
            self.frame_cnt = self.frame_cnt + 1
            self.saver.save_input_state_to_csv(obs["qpos"])
            self.saver.save_images_to_folder(obs["images"], frame_id=self.frame_cnt)
        print("main obs")
        return {
            "state": obs["qpos"],
            "images": obs["images"],
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        if self.save_obs and "actions" in action:
            self.saver.save_output_action_to_csv(action["actions"])
            print(f"action: {action['actions']}")
        print("main action apply")
        stop_flag = action.get("STOP", False)
        print(f"STOP_SIGNAL: {stop_flag}")
        if "actions" in action:
            raw = action["actions"]
            raw_array = np.asarray(raw)
            if raw_array.ndim == 0:
                truncated = raw_array
            elif raw_array.ndim == 1:
                truncated = raw_array[:14]
            else:
                truncated = raw_array[0, :14]
            self._ts = self._env.step(truncated, STOP=stop_flag)
        else:
            self._ts = self._env.step(None, STOP=stop_flag)
