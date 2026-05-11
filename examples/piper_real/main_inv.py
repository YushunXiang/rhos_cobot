# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import logging
import tyro
import rospy

from openpi_client import action_chunk_broker
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from rhos_cobot.openpi_remote_policy import create_resettable_websocket_policy

try:
    from examples.piper_real import env_inv as _env
    from examples.piper_real import logger as _logger
except ModuleNotFoundError:
    import env_inv as _env
    import logger as _logger


@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2" #H100
    port: int = 9000

    action_horizon: int = 32

    num_episodes: int = 1
    max_episode_steps: int = 1000
    
    save_log: bool = False


def main(args: Args) -> None:
    ws_client_policy = create_resettable_websocket_policy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    metadata = ws_client_policy.get_server_metadata()
    
    if args.save_log:
        # rospy.init_node('data_logger_node', anonymous=True)
        # input_img_logger = _logger.InputImgLogger()
        input_joint_state_logger = _logger.InputJointStateLogger()
        output_joint_state_logger = _logger.OutputJointStateLogger()
    
    runtime = _runtime.Runtime(
        environment=_env.PiperRealEnvironment(reset_position=metadata.get("reset_pose")),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
