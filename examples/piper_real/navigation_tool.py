import dataclasses
import logging
import time
from typing import Any
from typing import Callable

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


def _run_navigation_routine(
    prompt: str,
    *,
    routine_name: str,
    routine: tuple[tuple[float, float, float], ...],
    execute_step: Callable[[tuple[float, float, float]], None],
    stop_base_fn: Callable[[], None],
    dry_run: bool = False,
    inter_step_sleep_s: float = INTER_STEP_SLEEP_S,
) -> NavigationResult:
    logging.info(
        "Navigation tool invoked: prompt=%s routine=%s dry_run=%s",
        prompt,
        routine_name,
        dry_run,
    )

    if dry_run:
        return NavigationResult(
            ok=True,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=0,
        )

    executed_steps = 0
    try:
        for idx, step in enumerate(routine, start=1):
            logging.info(
                "Navigation step %d/%d: linear_x=%s angular_z=%s duration=%s",
                idx,
                len(routine),
                step[0],
                step[1],
                step[2],
            )
            execute_step(step)
            executed_steps = idx
            if inter_step_sleep_s > 0 and idx < len(routine):
                time.sleep(inter_step_sleep_s)
    except Exception as exc:  # noqa: BLE001
        stop_base_fn()
        logging.error("Navigation tool failed after %d steps: %s", executed_steps, exc)
        return NavigationResult(
            ok=False,
            prompt=prompt,
            routine_name=routine_name,
            executed_steps=executed_steps,
            error=str(exc),
        )

    stop_base_fn()
    return NavigationResult(
        ok=True,
        prompt=prompt,
        routine_name=routine_name,
        executed_steps=executed_steps,
    )


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
    if ros_operator is None:
        if not dry_run:
            return NavigationResult(
                ok=False,
                prompt=prompt,
                routine_name=DEFAULT_ROUTINE_NAME,
                executed_steps=0,
                error="ros_operator is required when dry_run is False",
            )

        stop_base_fn = lambda: None
        execute_step = lambda _step: None
    else:
        stop_base_fn = lambda: base_safety.stop_base(ros_operator)
        execute_step = lambda step: _execute_step(ros_operator, step)

    return _run_navigation_routine(
        prompt,
        routine_name=DEFAULT_ROUTINE_NAME,
        routine=DEFAULT_DEMO_ROUTINE,
        execute_step=execute_step,
        stop_base_fn=stop_base_fn,
        dry_run=dry_run,
        inter_step_sleep_s=INTER_STEP_SLEEP_S,
    )
