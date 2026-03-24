# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import logging

import tyro

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from examples.piper_real import env as _env
from examples.piper_real import logger as _logger
from examples.piper_real import llm_planner as _llm_planner
from examples.piper_real.planner_config import PlannerConfig


@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"  # H100
    port: int = 9000
    action_horizon: int = 16
    num_episodes: int = 1
    max_episode_steps: int = 1000
    save_log: bool = False
    prompt: str = ""
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)


def main(args: Args) -> None:
    args.planner.validate()

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    if args.save_log:
        _logger.InputJointStateLogger()
        _logger.OutputJointStateLogger()

    environment = _env.PiperRealEnvironment(
        reset_position=metadata.get("reset_pose"),
        prompt=args.prompt,
    )

    prompt = args.prompt.strip()
    if args.planner.enable_navigation:
        if prompt:
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)
            if not planner.confirm_navigation_safety(prompt):
                logging.error("Navigation aborted before execution; manipulation will not start.")
                return
            if not planner.run(task_prompt=prompt):
                logging.error("Navigation failed; manipulation will not start.")
                return
            logging.info("Navigation succeeded; starting manipulation runtime.")
        else:
            logging.info("Navigation enabled but prompt is empty; skipping navigation stage.")
            logging.info('Navigation status: {"status": "navigation_skipped", "reason": "empty prompt"}')
            logging.info("Navigation skipped; starting manipulation runtime.")
    else:
        logging.info("Navigation skipped because planner.enable_navigation is false.")
        logging.info('Navigation status: {"status": "navigation_skipped", "reason": "planner.enable_navigation is false"}')
        logging.info("Navigation skipped; starting manipulation runtime.")

    runtime = _runtime.Runtime(
        environment=environment,
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
