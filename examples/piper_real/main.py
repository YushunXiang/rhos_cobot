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
    navigation_only: bool = False  # Run navigation only, skip manipulation
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)


def main(args: Args) -> None:
    prompt = args.prompt.strip()

    if args.navigation_only and not args.use_llm_planner:
        logging.error("--navigation-only requires --use-llm-planner.")
        return

    if args.use_robot_base and not args.use_llm_planner:
        logging.error("--use-robot-base requires --use-llm-planner.")
        return

    if args.navigation_only and args.replay_dataset:
        logging.error("--navigation-only and --replay-dataset are mutually exclusive.")
        return

    if args.use_llm_planner and args.replay_dataset:
        logging.error("--use-llm-planner and --replay-dataset are mutually exclusive.")
        return

    # ── Replay mode: pure offline HDF5 data inspection ────────────────
    if args.replay_dataset:
        import h5py

        logging.info("Replay mode (offline): loading %s", args.replay_dataset)
        with h5py.File(args.replay_dataset, "r") as f:
            actions: np.ndarray = f["/action"][()]
            has_base_action = "base_action" in f
            base_actions: np.ndarray | None = (
                f["/base_action"][()] if has_base_action else None
            )

        num_steps = actions.shape[0]
        action_dim = actions.shape[1] if actions.ndim > 1 else 0

        for i in range(num_steps):
            arm = actions[i]
            arm_str = ", ".join(f"{v:.4f}" for v in arm[:14])
            if base_actions is not None and i < len(base_actions):
                base = base_actions[i]
                base_str = f"[{base[0]:.4f}, {base[1]:.4f}]"
            elif action_dim >= 16:
                base_str = f"[{arm[14]:.4f}, {arm[15]:.4f}]"
            else:
                base_str = "N/A"
            logging.info(
                "Replay step %d/%d -- arm: [%s] base: %s",
                i, num_steps, arm_str, base_str,
            )

        logging.info(
            "Replay complete: %d steps, action_dim=%d, has_base_action=%s",
            num_steps, action_dim, has_base_action,
        )
        return

    navigation_requested = args.use_llm_planner and bool(prompt)
    base_motion_requested = args.use_robot_base or navigation_requested

    if base_motion_requested:
        args.planner.validate_motion_limits()
    if navigation_requested:
        args.planner.validate_service_config()

    metadata = {}
    if not args.navigation_only:
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

    if args.navigation_only:
        logging.info("--navigation-only: skipping manipulation runtime.")
        _base_safety.stop_base(environment.ros_operator)
        return

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
    main(tyro.cli(Args))
