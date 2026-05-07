# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import logging
import tyro
import rospy

try:
    from examples.piper_real import qt_env as _qt_env
except ModuleNotFoundError:
    import qt_env as _qt_env

# Before env (cv2) / matplotlib Qt: avoid OpenCV's cv2/qt/plugins shadowing PySide6 xcb.
_qt_env.fix_qt_for_matplotlib()

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent

try:
    from examples.piper_real import env as _env
    from examples.piper_real import logger as _logger
except ModuleNotFoundError:
    import env as _env
    import logger as _logger


@dataclasses.dataclass
class Args:
    host: str = "10.42.0.2"  # H100
    port: int = 9000

    action_horizon: int = 16

    num_episodes: int = 1
    max_episode_steps: int = 5000

    save_log: bool = False

    prompt: str = ""

    # If True: each step log progress/task_progress/subtask_progress from action (INFO) when present
    # (dual-progress: keys come from server Policy.infer, not the checkpoint file alone). on_step
    # also logs full action keys (do not use env prints of action["actions"] as progress signal).
    log_progress: bool = False

    # Matplotlib live plot (task + subtask); set DISPLAY or SSH -X. Does not need log_progress.
    plot_progress: bool = False
    plot_progress_window: int = 500
    plot_progress_update_every: int = 1  # redraw at most every N control steps (1 = every step)


class _ProgressLogSubscriber(_subscriber.Subscriber):
    def on_episode_start(self) -> None:
        return

    def on_episode_end(self) -> None:
        return

    def on_step(self, observation: dict, action: dict) -> None:
        # PiperRealEnv calls rospy.init_node before the first step; ROS often elevates the root
        # logger so logging.info is silenced. Use print (same as env "main obs") for visibility.
        print(f"on_step action keys={sorted(action.keys())}", flush=True)
        for key in ("progress", "task_progress", "subtask_progress"):
            if key not in action:
                continue
            if action[key] is not None:
                print(f"policy {key} = {action[key]}", flush=True)
            else:
                print(
                    f"policy {key} key present but filtered (None); repr={action.get(key)!r}",
                    flush=True,
                )


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
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

    logging.info("log_progress=%s", getattr(args, "log_progress", "<missing>"))
    subscribers = []
    if args.log_progress:
        subscribers.append(_ProgressLogSubscriber())
    if args.plot_progress:
        try:
            from examples.piper_real import progress_plot as _progress_plot
        except ModuleNotFoundError:
            import progress_plot as _progress_plot
        subscribers.append(
            _progress_plot.DualProgressPlotSubscriber(
                window=args.plot_progress_window,
                update_every=args.plot_progress_update_every,
            )
        )

    runtime = _runtime.Runtime(
        environment=_env.PiperRealEnvironment(reset_position=metadata.get("reset_pose"), prompt=args.prompt),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=subscribers,
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # rospy (initialized inside PiperRealEnvironment) may reconfigure loggers (often root->WARNING);
    # re-assert INFO on root and each handler so later logging.info is not silenced.
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    for _h in _root.handlers:
        _h.setLevel(logging.INFO)

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)