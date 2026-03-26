import logging
import math
from typing import Any


TRACER_MANUAL_MAX_LINEAR_VEL_MPS = 1.8
TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S = 1.0


def confirm_base_motion_safety(
    task_prompt: str,
    *,
    use_llm_planner: bool,
    use_robot_base: bool,
) -> bool:
    requested_modes: list[str] = []
    if use_llm_planner:
        requested_modes.append("LLM navigation")
    if use_robot_base:
        requested_modes.append("policy-driven base control")

    logging.warning(
        "TRACER base-motion safety confirmation required. "
        "Review docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf before continuing."
    )
    logging.warning(
        "Manual preflight: use only in a clear visible area, keep the robot within sight, "
        "verify both emergency stops are released, verify battery voltage is above 22.5V, "
        "and do not rely on autonomous obstacle avoidance."
    )
    if requested_modes:
        logging.info("Requested base-motion modes: %s", ", ".join(requested_modes))
    logging.info("Planned task: %s", task_prompt or "<empty prompt>")
    try:
        answer = input("Type 'yes' to allow TRACER base motion for this run: ").strip().lower()
    except EOFError:
        answer = ""

    confirmed = answer == "yes"
    logging.info('Base motion confirmation: {"confirmed": %s}', str(confirmed).lower())
    return confirmed


def enforce_base_velocity_limits(
    linear_x: float,
    angular_z: float,
    *,
    max_linear_vel: float,
    max_angular_vel: float,
    source: str,
) -> tuple[float, float]:
    if not math.isfinite(linear_x):
        raise ValueError(f"{source} linear velocity must be finite")
    if not math.isfinite(angular_z):
        raise ValueError(f"{source} angular velocity must be finite")
    if abs(linear_x) > max_linear_vel:
        raise ValueError(f"{source} linear velocity {linear_x} exceeds limit {max_linear_vel}")
    if abs(angular_z) > max_angular_vel:
        raise ValueError(f"{source} angular velocity {angular_z} exceeds limit {max_angular_vel}")
    return linear_x, angular_z


def stop_base(ros_operator: Any) -> None:
    ros_operator.robot_base_publish([0.0, 0.0])
