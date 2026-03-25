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
from examples.piper_real import task_decomposer as _task_decomposer


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

    # ── Stationary manipulation (no LLM planner) ─────────────────────
    if not args.use_llm_planner:
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
        return

    # ── Two-layer LLM planner ────────────────────────────────────────
    args.planner.validate_service_config()

    if not prompt:
        logging.info("LLM planner enabled but prompt is empty; running stationary manipulation.")
        # Fall through — create server connection and run a single manipulation episode
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
        return

    # Step 0: Validate motion limits early to avoid wasting an LLM call
    if args.use_robot_base:
        args.planner.validate_motion_limits()

    # Step 1: Decompose task
    decomposer = _task_decomposer.TaskDecomposer(args.planner)
    try:
        subtask_list = decomposer.decompose(prompt)
    except _task_decomposer.DecompositionError as exc:
        logging.error("Task decomposition failed: %s", exc)
        return

    has_navigate = any(s.type == "navigate" for s in subtask_list)
    has_manipulate = any(s.type == "manipulate" for s in subtask_list)
    needs_server = has_manipulate and not args.navigation_only

    # Step 2: Safety confirmation (once, if base motion requested)
    if args.use_robot_base and has_navigate:
        if not _base_safety.confirm_base_motion_safety(
            prompt,
            use_llm_planner=True,
            use_robot_base=False,  # pass False to suppress misleading "policy-driven base control" label
        ):
            logging.error("Base motion aborted before execution.")
            return

    # Step 3: Create inference server connection if needed
    ws_client_policy = None
    metadata = {}
    if needs_server:
        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

    # Step 4: Create shared environment if needed
    environment = None
    if needs_server:
        if args.save_log:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=prompt,
        )

    # Step 5: Create navigation planner if needed
    planner = None
    if args.use_robot_base and has_navigate:
        if environment is not None:
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)
        else:
            # Navigation-only mode: need ROS for base movement but no manipulation env
            # Create a minimal environment just for ros_operator access
            environment = _env.PiperRealEnvironment(
                reset_position=None,
                prompt=prompt,
            )
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)

    # Step 6: Execute subtask loop
    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing subtask %d/%d [%s]: %s",
                idx + 1, len(subtask_list), subtask.type, subtask.prompt,
            )

            if subtask.type == "navigate":
                if args.use_robot_base:
                    if not planner.run(task_prompt=subtask.prompt):
                        _base_safety.stop_base(environment.ros_operator)
                        logging.error(
                            "Navigation failed at subtask %d/%d; aborting.",
                            idx + 1, len(subtask_list),
                        )
                        return
                    logging.info("Navigate subtask %d/%d succeeded.", idx + 1, len(subtask_list))
                else:
                    logging.info("Navigate (dry-run): %s", subtask.prompt)

            elif subtask.type == "manipulate":
                if args.navigation_only:
                    logging.info("Manipulate (skipped): %s", subtask.prompt)
                    continue

                assert ws_client_policy is not None, "manipulate subtask requires server connection"
                environment.set_prompt(subtask.prompt)
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
                logging.info("Manipulate subtask %d/%d completed.", idx + 1, len(subtask_list))

        logging.info("All subtasks completed successfully.")
    finally:
        if args.use_robot_base and environment is not None:
            _base_safety.stop_base(environment.ros_operator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
