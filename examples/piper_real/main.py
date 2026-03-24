# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import logging

import numpy as np

import tyro

from examples.piper_real import base_safety as _base_safety
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
    replay_dataset: str = ""  # Path to HDF5 episode file for offline replay
    use_llm_planner: bool = False
    use_robot_base: bool = False
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)


def main(args: Args) -> None:
    prompt = args.prompt.strip()

    # ── Replay mode: skip ROS, safety, navigation ──────────────────────
    if args.replay_dataset:
        from examples.piper_real import replay_env as _replay_env

        logging.info("Replay mode: loading %s", args.replay_dataset)
        environment = _replay_env.ReplayEnvironment(
            dataset_path=args.replay_dataset,
            prompt=prompt,
        )

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

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

        # ── Post-replay summary ────────────────────────────────────────
        if environment.predicted_actions:
            predicted = np.stack(environment.predicted_actions)
            gt = environment.ground_truth_actions[: len(predicted)]
            mae = np.mean(np.abs(predicted[:, :14] - gt[:, :14]))
            logging.info(
                "Replay finished: %d steps, MAE vs ground-truth: %.6f",
                len(predicted),
                mae,
            )
        return

    navigation_requested = args.use_llm_planner and bool(prompt)
    base_motion_requested = args.use_robot_base or navigation_requested

    if base_motion_requested:
        args.planner.validate_motion_limits()
    if navigation_requested:
        args.planner.validate_service_config()

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
        use_robot_base=args.use_robot_base,
        max_base_linear_vel=args.planner.max_linear_vel,
        max_base_angular_vel=args.planner.max_angular_vel,
    )

    if base_motion_requested and not _base_safety.confirm_base_motion_safety(
        prompt,
        use_llm_planner=navigation_requested,
        use_robot_base=args.use_robot_base,
    ):
        _base_safety.stop_base(environment.ros_operator)
        logging.error("Base motion aborted before execution; manipulation will not start.")
        return

    if args.use_llm_planner:
        if prompt:
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)
            if not planner.run(task_prompt=prompt):
                _base_safety.stop_base(environment.ros_operator)
                logging.error("Navigation failed; manipulation will not start.")
                return
            logging.info("Navigation succeeded; starting manipulation runtime.")
        else:
            logging.info("Navigation enabled but prompt is empty; skipping navigation stage.")
            logging.info('Navigation status: {"status": "navigation_skipped", "reason": "empty prompt"}')
            logging.info("Navigation skipped; starting manipulation runtime.")
    else:
        logging.info("Navigation skipped because use_llm_planner is false.")
        logging.info('Navigation status: {"status": "navigation_skipped", "reason": "use_llm_planner is false"}')
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
    try:
        runtime.run()
    finally:
        if args.use_llm_planner or args.use_robot_base:
            _base_safety.stop_base(environment.ros_operator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
