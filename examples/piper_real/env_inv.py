
from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    from examples.piper_real import real_env as _real_env
    from examples.piper_real.logger import ModelInputObservationSaver as _obs_saver
except ModuleNotFoundError:
    import real_env as _real_env
    from logger import ModelInputObservationSaver as _obs_saver


class PiperRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None
        self.save_obs = True
        self.frame_cnt = 0
        if self.save_obs:
            self.saver = _obs_saver()

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

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

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            
        #normalization for qpos puppet gript: TODO

        # 保存观察结果
        if self.save_obs:
            self.frame_cnt = self.frame_cnt+1
            self.saver.save_input_state_to_csv(obs["qpos"])
            self.saver.save_images_to_folder(obs["images"],frame_id=self.frame_cnt)
        
        import numpy as np
        return {
            "state": np.concatenate([obs["qpos"][7:14],obs["qpos"][0:7],obs["qpos"][14:]],axis=0),
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        if self.save_obs:
            self.saver.save_output_action_to_csv(action["actions"])

        print(f"action: {action['actions']}")
        import numpy as np
        self._ts = self._env.step(np.concatenate([action["actions"][7:14],action["actions"][0:7],action["actions"][14:]],axis=0))





