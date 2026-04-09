# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import logging

import numpy as np

import tyro

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime.agents import policy_agent as _policy_agent
from examples.piper_real.planner_config import PlannerConfig


DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_REPLAY_MANIPULATE_MAX_STEPS = 64
DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS = 16
_VALID_REPLAY_MODES = {"policy", "planner", "hybrid"}

@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"  # H100
    port: int = 9000
    action_horizon: int = 16
    num_episodes: int = 1
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    save_log: bool = False
    prompt: str = ""
    replay_dataset: str = ""  # Path to HDF5 episode file for offline replay
    replay_mode: str = "policy"  # policy | planner | hybrid
    replay_manipulate_max_steps: int = DEFAULT_REPLAY_MANIPULATE_MAX_STEPS
    replay_manipulate_replan_interval_steps: int = DEFAULT_REPLAY_MANIPULATE_REPLAN_INTERVAL_STEPS
    use_llm_planner: bool = False
    use_robot_base: bool = False
    navigation_only: bool = False  # Run navigation only, skip manipulation
    skip_server_checks: bool = False
    server_check_timeout_sec: float = 5.0
    visualize: bool = False  # Show camera views + subtask overlay during replay (web UI)
    visualize_port: int = 7860  # Port for the visualization web server
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)


def _create_policy_agent(args: Args) -> _policy_agent.PolicyAgent:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    return _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )
    )


def _run_required_server_checks(
    args: Args,
    *,
    needs_pi0: bool = False,
    needs_planner: bool = False,
) -> bool:
    if args.skip_server_checks or (not needs_pi0 and not needs_planner):
        return True

    from examples.piper_real import server_checks as _server_checks

    try:
        if needs_planner:
            _server_checks.check_planner_server(
                args.planner.base_url,
                expected_model=args.planner.model,
                timeout_sec=args.server_check_timeout_sec,
            )
        if needs_pi0:
            _server_checks.check_pi0_server(
                args.host,
                args.port,
                timeout_sec=args.server_check_timeout_sec,
            )
    except _server_checks.ServerCheckError as exc:
        logging.error("%s", exc)
        return False

    return True


def _log_replay_summary(environment, executed_steps: int) -> None:
    if not environment.predicted_actions:
        logging.warning("Replay finished without any predicted actions.")
        return

    predicted = np.stack(environment.predicted_actions)
    gt_indices = np.asarray(environment.predicted_action_steps, dtype=np.int64)
    gt = environment.ground_truth_actions[gt_indices]

    arm_dim = min(14, predicted.shape[-1], gt.shape[-1])
    arm_mae = float(np.mean(np.abs(predicted[:, :arm_dim] - gt[:, :arm_dim])))

    base_suffix = ", base_mae=N/A"
    if predicted.shape[-1] >= 16:
        if environment.ground_truth_base_actions is not None:
            gt_base = environment.ground_truth_base_actions[gt_indices]
            base_mae = float(np.mean(np.abs(predicted[:, 14:16] - gt_base)))
            base_suffix = f", base_mae={base_mae:.6f}"
        elif gt.shape[-1] >= 16:
            gt_base = gt[:, 14:16]
            base_mae = float(np.mean(np.abs(predicted[:, 14:16] - gt_base)))
            base_suffix = f", base_mae={base_mae:.6f}"

    logging.info(
        "Replay finished: executed_steps=%d/%d, predicted_steps=%d, arm_mae=%.6f%s",
        executed_steps,
        environment.num_steps,
        len(predicted),
        arm_mae,
        base_suffix,
    )


def _resolve_replay_mode(args: Args) -> str:
    replay_mode = args.replay_mode.strip().lower()
    if replay_mode not in _VALID_REPLAY_MODES:
        raise ValueError(
            f"--replay-mode must be one of {sorted(_VALID_REPLAY_MODES)}, got {args.replay_mode!r}."
        )
    if args.use_llm_planner and replay_mode == "policy":
        logging.info("Replay mode inferred as planner from legacy --use-llm-planner flag.")
        return "planner"
    return replay_mode


def _build_replay_subtask_list(args: Args, prompt: str):
    from examples.piper_real import task_decomposer as _task_decomposer

    decomposer = _task_decomposer.TaskDecomposer(args.planner)
    try:
        subtask_list = decomposer.decompose(prompt)
    except _task_decomposer.DecompositionError as exc:
        if args.navigation_only:
            logging.warning(
                "Replay decomposition failed in navigation-only mode (%s); "
                "using the original prompt as a single navigate subtask.",
                exc,
            )
            return [_task_decomposer.Subtask(type="navigate", prompt=prompt)]
        logging.error("Task decomposition failed: %s", exc)
        return None

    if args.navigation_only and not any(subtask.type == "navigate" for subtask in subtask_list):
        logging.warning(
            "Replay decomposition returned no navigate subtasks; "
            "using the original prompt as a single navigate subtask."
        )
        return [_task_decomposer.Subtask(type="navigate", prompt=prompt)]

    return subtask_list


def _run_replay_manipulation_subtask(
    environment,
    agent: _policy_agent.PolicyAgent,
    manipulation_planner,
    *,
    subtask_prompt: str,
    max_steps: int,
    replan_interval_steps: int,
    visualizer=None,
) -> dict[str, object]:
    prompt_history: list[dict[str, object]] = []
    current_policy_prompt = subtask_prompt
    prompt_queries = 0

    def _replan(policy_steps: int) -> bool:
        nonlocal current_policy_prompt
        nonlocal prompt_queries
        try:
            decision = manipulation_planner.plan(
                task_prompt=subtask_prompt,
                current_policy_prompt=current_policy_prompt,
                executed_policy_steps=policy_steps,
                prompt_history=prompt_history,
            )
            prompt_queries += 1
        except Exception as exc:  # noqa: BLE001
            environment.set_prompt(current_policy_prompt)
            agent.reset()
            prompt_history.append(
                {
                    "policy_steps": policy_steps,
                    "prompt": current_policy_prompt,
                    "reason": f"replan_failed: {exc}",
                }
            )
            logging.error(
                "Replay manipulate replanner failed after %d policy steps; reusing current prompt: %s",
                policy_steps,
                exc,
            )
            return True

        if decision.action == "complete":
            logging.info(
                "Replay manipulate replanner marked subtask complete after %d policy steps: %s",
                policy_steps,
                decision.reason,
            )
            return False

        current_policy_prompt = decision.prompt
        prompt_history.append(
            {
                "policy_steps": policy_steps,
                "prompt": decision.prompt,
                "reason": decision.reason,
            }
        )
        environment.set_prompt(decision.prompt)
        agent.reset()
        logging.info(
            "Replay manipulate replanner prompt update after %d policy steps: %s",
            policy_steps,
            decision.prompt,
        )
        return True

    if not _replan(0):
        return {
            "executed_steps": 0,
            "prompt_queries": prompt_queries,
            "completed_by_replan": True,
            "last_policy_prompt": current_policy_prompt,
        }

    executed_steps = 0
    while executed_steps < max_steps and not environment.is_episode_complete():
        observation = environment.get_observation()
        action = agent.get_action(observation)
        environment.apply_action(action)
        executed_steps += 1

        if visualizer is not None:
            step_idx = environment.get_cursor() - 1
            if not visualizer.update(step_idx, extra_info=f"policy prompt: {current_policy_prompt}"):
                logging.info("Replay manipulate aborted by user at policy step %d.", executed_steps)
                break

        if (
            executed_steps < max_steps
            and not environment.is_episode_complete()
            and executed_steps % replan_interval_steps == 0
            and not _replan(executed_steps)
        ):
            return {
                "executed_steps": executed_steps,
                "prompt_queries": prompt_queries,
                "completed_by_replan": True,
                "last_policy_prompt": current_policy_prompt,
            }

    if executed_steps >= max_steps and not environment.is_episode_complete():
        logging.info("Replay manipulate subtask hit fixed cap at %d policy steps.", max_steps)
    elif environment.is_episode_complete():
        logging.info("Replay manipulate subtask stopped because the replay dataset is exhausted.")

    return {
        "executed_steps": executed_steps,
        "prompt_queries": prompt_queries,
        "completed_by_replan": False,
        "last_policy_prompt": current_policy_prompt,
    }


def _log_hybrid_replay_summary(
    environment,
    *,
    total_subtasks: int,
    navigate_subtasks: int,
    manipulate_subtasks: int,
    policy_steps: int,
    prompt_queries: int,
) -> None:
    logging.info(
        "Hybrid replay completed: subtasks=%d, navigate=%d, manipulate=%d, "
        "policy_steps=%d, prompt_queries=%d, replay_cursor=%d/%d",
        total_subtasks,
        navigate_subtasks,
        manipulate_subtasks,
        policy_steps,
        prompt_queries,
        environment.get_cursor(),
        environment.num_steps,
    )
    if policy_steps > 0:
        _log_replay_summary(environment, policy_steps)


def _run_replay_inference(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error("--replay-dataset currently supports only --num-episodes=1.")
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay mode.")

    logging.info("Replay mock mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
    )

    if not _run_required_server_checks(args, needs_pi0=True):
        return

    agent = _create_policy_agent(args)

    if args.max_episode_steps == DEFAULT_MAX_EPISODE_STEPS and environment.num_steps > args.max_episode_steps:
        logging.warning(
            "Replay dataset has %d steps; default --max-episode-steps=%d will truncate it. "
            "Pass --max-episode-steps 0 to run the full dataset.",
            environment.num_steps,
            args.max_episode_steps,
        )

    visualizer = _replay_visualizer.ReplayVisualizer(environment, enabled=args.visualize, port=args.visualize_port)
    visualizer.set_subtask_context(1, 1, "policy", prompt)

    environment.reset()
    agent.reset()

    executed_steps = 0
    try:
        while not environment.is_episode_complete():
            observation = environment.get_observation()
            action = agent.get_action(observation)
            environment.apply_action(action)
            executed_steps += 1

            if not visualizer.update(executed_steps - 1):
                logging.info("Replay aborted by user at step %d.", executed_steps)
                break

            if args.max_episode_steps > 0 and executed_steps >= args.max_episode_steps:
                logging.info(
                    "Replay stopped early at %d steps due to --max-episode-steps=%d.",
                    executed_steps,
                    args.max_episode_steps,
                )
                break
    finally:
        visualizer.close()

    _log_replay_summary(environment, executed_steps)


def _run_replay_planner(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import replay_planner as _replay_planner
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error("--replay-dataset with --use-llm-planner currently supports only --num-episodes=1.")
        return

    if not prompt:
        logging.error("--use-llm-planner with --replay-dataset requires a non-empty --prompt.")
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay planner mode.")

    args.planner.validate_service_config()
    args.planner.validate_motion_limits()
    if not _run_required_server_checks(args, needs_planner=True):
        return

    logging.info("Replay planner mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
        max_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    if not args.navigation_only:
        logging.warning(
            "Replay planner mode evaluates navigate subtasks only; manipulate subtasks will be skipped."
        )

    subtask_list = _build_replay_subtask_list(args, prompt)
    if subtask_list is None:
        environment.close()
        return

    visualizer = _replay_visualizer.ReplayVisualizer(environment, enabled=args.visualize, port=args.visualize_port)

    def _on_nav_step(step_idx: int) -> bool:
        return visualizer.update(step_idx)

    planner = _replay_planner.OfflineReplayNavigationPlanner(
        environment, args.planner, on_step_callback=_on_nav_step,
    )
    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing replay subtask %d/%d [%s]: %s",
                idx + 1, len(subtask_list), subtask.type, subtask.prompt,
            )

            if subtask.type == "navigate":
                visualizer.set_subtask_context(idx + 1, len(subtask_list), subtask.type, subtask.prompt)
                if not planner.run(task_prompt=subtask.prompt):
                    logging.error(
                        "Replay navigation failed at subtask %d/%d; aborting.",
                        idx + 1, len(subtask_list),
                    )
                    return
                logging.info(
                    "Replay navigate subtask %d/%d succeeded at replay step %d/%d.",
                    idx + 1,
                    len(subtask_list),
                    planner.current_step,
                    environment.num_steps,
                )
            elif args.navigation_only:
                logging.info("Manipulate (skipped): %s", subtask.prompt)
            else:
                logging.info("Manipulate (skipped in replay planner mode): %s", subtask.prompt)

        logging.info(
            "Replay planner completed successfully at replay step %d/%d using camera %s.",
            planner.current_step,
            environment.num_steps,
            environment.front_camera_name,
        )
    finally:
        visualizer.close()
        environment.close()


def _run_replay_hybrid(args: Args, prompt: str) -> None:
    from examples.piper_real import replay_env as _replay_env
    from examples.piper_real import replay_manipulation_planner as _replay_manipulation_planner
    from examples.piper_real import replay_planner as _replay_planner
    from examples.piper_real import replay_visualizer as _replay_visualizer

    if args.num_episodes != 1:
        logging.error("--replay-dataset with --replay-mode=hybrid currently supports only --num-episodes=1.")
        return

    if not prompt:
        logging.error("--replay-mode=hybrid with --replay-dataset requires a non-empty --prompt.")
        return

    if args.replay_manipulate_max_steps <= 0:
        logging.error("--replay-manipulate-max-steps must be positive in hybrid replay mode.")
        return

    if args.replay_manipulate_replan_interval_steps <= 0:
        logging.error("--replay-manipulate-replan-interval-steps must be positive in hybrid replay mode.")
        return

    if args.save_log:
        logging.info("--save-log is ignored in replay hybrid mode.")

    args.planner.validate_service_config()
    args.planner.validate_motion_limits()
    if not _run_required_server_checks(args, needs_pi0=not args.navigation_only, needs_planner=True):
        return

    logging.info("Replay hybrid mode: loading %s", args.replay_dataset)
    environment = _replay_env.ReplayEnvironment(
        dataset_path=args.replay_dataset,
        prompt=prompt,
        max_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    subtask_list = _build_replay_subtask_list(args, prompt)
    if subtask_list is None:
        environment.close()
        return

    visualizer = _replay_visualizer.ReplayVisualizer(environment, enabled=args.visualize, port=args.visualize_port)

    def _on_nav_step(step_idx: int) -> bool:
        return visualizer.update(step_idx)

    planner = _replay_planner.OfflineReplayNavigationPlanner(
        environment, args.planner, on_step_callback=_on_nav_step,
    )
    policy_agent = None if args.navigation_only else _create_policy_agent(args)
    manipulation_planner = (
        None
        if args.navigation_only
        else _replay_manipulation_planner.ReplayManipulationPromptPlanner(environment, args.planner)
    )
    completed_navigate = 0
    completed_manipulate = 0
    policy_steps = 0
    prompt_queries = 0

    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing replay subtask %d/%d [%s]: %s",
                idx + 1,
                len(subtask_list),
                subtask.type,
                subtask.prompt,
            )
            visualizer.set_subtask_context(idx + 1, len(subtask_list), subtask.type, subtask.prompt)

            if subtask.type == "navigate":
                if not planner.run(task_prompt=subtask.prompt):
                    logging.error(
                        "Replay navigation failed at subtask %d/%d; aborting.",
                        idx + 1,
                        len(subtask_list),
                    )
                    return
                completed_navigate += 1
                logging.info(
                    "Replay navigate subtask %d/%d succeeded at replay step %d/%d.",
                    idx + 1,
                    len(subtask_list),
                    environment.get_cursor(),
                    environment.num_steps,
                )
                continue

            if args.navigation_only:
                logging.info("Manipulate (skipped): %s", subtask.prompt)
                continue

            assert policy_agent is not None, "hybrid manipulate subtask requires a policy agent"
            assert manipulation_planner is not None, "hybrid manipulate subtask requires a prompt replanner"
            manipulation_result = _run_replay_manipulation_subtask(
                environment,
                policy_agent,
                manipulation_planner,
                subtask_prompt=subtask.prompt,
                max_steps=args.replay_manipulate_max_steps,
                replan_interval_steps=args.replay_manipulate_replan_interval_steps,
                visualizer=visualizer,
            )
            executed_steps = int(manipulation_result["executed_steps"])
            completed_manipulate += 1
            policy_steps += executed_steps
            prompt_queries += int(manipulation_result["prompt_queries"])
            logging.info(
                "Replay manipulate subtask %d/%d completed after %d policy steps, "
                "%d prompt queries, at replay cursor %d/%d.",
                idx + 1,
                len(subtask_list),
                executed_steps,
                int(manipulation_result["prompt_queries"]),
                environment.get_cursor(),
                environment.num_steps,
            )
            if environment.is_episode_complete() and idx + 1 < len(subtask_list):
                logging.error(
                    "Replay dataset exhausted after subtask %d/%d; aborting remaining subtasks.",
                    idx + 1,
                    len(subtask_list),
                )
                return

        _log_hybrid_replay_summary(
            environment,
            total_subtasks=len(subtask_list),
            navigate_subtasks=completed_navigate,
            manipulate_subtasks=completed_manipulate,
            policy_steps=policy_steps,
            prompt_queries=prompt_queries,
        )
    finally:
        visualizer.close()
        environment.close()


def main(args: Args) -> None:
    prompt = args.prompt.strip()

    if args.replay_dataset:
        try:
            replay_mode = _resolve_replay_mode(args)
        except ValueError as exc:
            logging.error("%s", exc)
            return

        if args.use_robot_base:
            logging.error("--use-robot-base and --replay-dataset are mutually exclusive.")
            return

        if replay_mode == "planner":
            _run_replay_planner(args, prompt)
            return

        if replay_mode == "hybrid":
            _run_replay_hybrid(args, prompt)
            return

        if args.navigation_only:
            logging.error(
                "--navigation-only requires replay planner mode "
                "(--replay-mode planner|hybrid or legacy --use-llm-planner)."
            )
            return

        _run_replay_inference(args, prompt)
        return

    if args.navigation_only and not args.use_llm_planner:
        logging.error("--navigation-only requires --use-llm-planner.")
        return

    if args.use_robot_base and not args.use_llm_planner:
        logging.error("--use-robot-base requires --use-llm-planner.")
        return

    from examples.piper_real import base_safety as _base_safety
    from examples.piper_real import env as _env
    from examples.piper_real import logger as _logger
    from examples.piper_real import navigation_tool as _navigation_tool
    from examples.piper_real import task_decomposer as _task_decomposer
    from openpi_client.runtime import runtime as _runtime

    # ── Stationary manipulation (no LLM planner, or LLM planner with empty prompt) ──
    if not args.use_llm_planner or not prompt:
        if args.use_llm_planner:
            args.planner.validate_service_config()
            logging.info("LLM planner enabled but prompt is empty; running stationary manipulation.")

        if not _run_required_server_checks(args, needs_pi0=True):
            return

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
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

    if not _run_required_server_checks(args, needs_planner=True):
        return

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
    needs_ros_environment = needs_server or (args.use_robot_base and has_navigate)

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
        if not _run_required_server_checks(args, needs_pi0=True):
            return

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

    # Step 4: Create shared environment if needed
    environment = None
    if needs_ros_environment:
        if args.save_log and needs_server:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose") if needs_server else None,
            prompt=prompt,
        )

    # Step 6: Execute subtask loop
    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing subtask %d/%d [%s]: %s",
                idx + 1, len(subtask_list), subtask.type, subtask.prompt,
            )

            if subtask.type == "navigate":
                ros_operator = None if environment is None else environment.ros_operator
                result = _navigation_tool.navigate(
                    subtask.prompt,
                    ros_operator,
                    dry_run=not args.use_robot_base,
                )
                if not result.ok:
                    logging.error(
                        "Navigation failed at subtask %d/%d: %s",
                        idx + 1, len(subtask_list), result.error or "unknown error",
                    )
                    return
                logging.info(
                    "Navigate subtask %d/%d succeeded via routine %s.",
                    idx + 1, len(subtask_list), result.routine_name,
                )

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
