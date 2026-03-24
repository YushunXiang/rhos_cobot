# Piper Real Deploy Flow

This example runs the real-robot deploy path for `examples/piper_real/main.py`.

The flow is now navigation-first:

1. Build `PiperRealEnvironment` and connect to the OpenPI policy server.
2. If navigation is enabled and `--prompt` is non-empty, query a local OpenAI-compatible vision planner with the front camera image and odometry.
3. Drive the TRACER base in short bounded cycles until the planner returns `stop`.
4. Start OpenPI manipulation with the original prompt only after navigation succeeds, or immediately when navigation is explicitly disabled.

## Safety

Before implementation, bring-up, or deploy validation, review the local manual:

- `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`

The runtime now requires explicit operator confirmation before any chassis motion. This is intentional: the TRACER manual documents that the platform does not provide built-in autonomous obstacle avoidance, anti-drop, or biological proximity warning functions.

## Python Environment

```bash
uv venv --python 3.11 examples/piper_real/.venv
source examples/piper_real/.venv/bin/activate
uv pip compile examples/piper_real/requirements.in -o examples/piper_real/requirements.txt --python-version 3.11
uv pip sync examples/piper_real/requirements.txt
```

`examples/piper_real/main.py` imports `openpi_client`. This repository snapshot does not contain `packages/openpi-client`, so use the existing deploy environment that already provides `openpi_client` on `PYTHONPATH` or in the active interpreter.

## Robot Workstation Setup

```bash
sh scripts/init.sh
conda activate aloha
init_deploy
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
source examples/piper_real/.venv/bin/activate
```

If needed, configure the robot workstation network first as described in [docs/deploy.md](../../docs/deploy.md).

## Local Planner Service

Start a local OpenAI-compatible multimodal planner service before launching the runtime.

Minimum requirements:

- Accepts image + text chat-completions requests.
- Returns JSON-only planner responses.
- Is reachable from the robot workstation at the configured `--planner.base-url`.

The planner must return one of these shapes:

```json
{"action": "move", "linear_x": 0.2, "angular_z": -0.1, "duration": 1.2, "reasoning": "Rotate slightly and move forward."}
```

```json
{"action": "stop", "reason": "The robot is in a usable operating position."}
```

## Run Navigation First

```bash
python -m examples.piper_real.main \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://localhost:8000/v1 \
  --planner.model qwen2.5-vl-72b
```

Runtime behavior:

- A safety warning is shown before navigation starts.
- The operator must type `yes` to allow base motion.
- Planner cycles keep the base stopped while waiting for a decision.
- Each accepted `move` command is executed with bounded velocities and followed by an immediate zero-velocity command.
- Manipulation starts only after a planner `stop` decision.

## Skip Navigation

```bash
python -m examples.piper_real.main \
  --prompt "拿起红色杯子" \
  --planner.enable-navigation false
```

This path skips chassis movement, reports `navigation skipped`, and starts manipulation directly.

## Validation

During bring-up, use:

```bash
rostopic echo /cmd_vel
```

Confirm all of the following:

- Navigation publishes `/cmd_vel` before arm manipulation starts.
- The base returns to zero velocity between planner cycles.
- Planner logs show each raw response, rejected or retried cycles, and the terminal reason.
- Navigation failure prevents `Runtime.run()` from starting.
