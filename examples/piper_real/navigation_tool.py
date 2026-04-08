import dataclasses
import logging
import time
from typing import Any

from examples.piper_real import base_safety


DEFAULT_CONTROL_HZ = 10.0
INTER_STEP_SLEEP_S = 1.0
DEFAULT_ROUTINE_NAME = "default_demo"
DEFAULT_DEMO_ROUTINE: tuple[tuple[float, float, float], ...] = (
    (-0.2, 0.0, 1.0),
    (0.0, 0.2, 9.0),
    (0.2, 0.0, 1.5),
    (0.0, -0.2, 9.0),
    (0.1, 0.0, 2.0),
)


@dataclasses.dataclass
class NavigationResult:
    ok: bool
    prompt: str
    routine_name: str
    executed_steps: int
    error: str | None = None


def _execute_step(
    ros_operator: Any,
    step: tuple[float, float, float],
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
) -> None:
    linear_x, angular_z, duration = step
    start = time.monotonic()
    period = 1.0 / control_hz
    next_tick = start

    while time.monotonic() - start < duration:
        ros_operator.robot_base_publish([linear_x, angular_z])
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()

    base_safety.stop_base(ros_operator)


def navigate(
    prompt: str,
    ros_operator: Any | None,
    *,
    dry_run: bool = False,
) -> NavigationResult:
    logging.info(
        "Navigation tool invoked: prompt=%s routine=%s dry_run=%s",
        prompt,
        DEFAULT_ROUTINE_NAME,
        dry_run,
    )

    if dry_run:
        return NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=0,
        )

    if ros_operator is None:
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=0,
            error="ros_operator is required when dry_run is False",
        )

    executed_steps = 0
    try:
        for idx, step in enumerate(DEFAULT_DEMO_ROUTINE, start=1):
            logging.info(
                "Navigation step %d/%d: linear_x=%s angular_z=%s duration=%s",
                idx,
                len(DEFAULT_DEMO_ROUTINE),
                step[0],
                step[1],
                step[2],
            )
            _execute_step(ros_operator, step)
            executed_steps = idx
            if idx < len(DEFAULT_DEMO_ROUTINE):
                time.sleep(INTER_STEP_SLEEP_S)
    except Exception as exc:  # noqa: BLE001
        base_safety.stop_base(ros_operator)
        logging.error("Navigation tool failed after %d steps: %s", executed_steps, exc)
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=DEFAULT_ROUTINE_NAME,
            executed_steps=executed_steps,
            error=str(exc),
        )

    base_safety.stop_base(ros_operator)
    return NavigationResult(
        ok=True,
        prompt=prompt,
        routine_name=DEFAULT_ROUTINE_NAME,
        executed_steps=executed_steps,
    )
