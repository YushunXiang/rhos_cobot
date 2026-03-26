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

Any run that can move the TRACER base now requires explicit operator confirmation before motion begins. This applies to both `--use-llm-planner` and `--use-robot-base`.

Shared constraints:

- Use only in a clear visible area.
- Verify both emergency stops are released.
- Keep battery voltage above the manual warning threshold of 22.5V.
- Do not exceed the configured velocity limits; startup rejects limits above the manual maximums of `1.8 m/s` linear and `1.0 rad/s` angular.

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

Before running this example, bring up the TRACER base bridge that publishes `/odom_raw`, consumes `/cmd_vel`, and keeps the lower-level CAN control alive within the platform's 500ms command timeout. If you use the official AgileX stack, the CAN side should match the manual's `gs_usb` + `can0` 500k setup and command-mode requirements.

## Local Planner Service

Start a local OpenAI-compatible multimodal planner service before launching the runtime.

For the default planner configuration in this repo, start the remote vLLM service with:

```bash
bash scripts/start_vllm_server.sh
```

This connects to `web@192.168.3.123`, starts `vllm serve` inside a remote `tmux` session, and serves `Qwen/Qwen3.5-4B` at the default planner endpoint `http://192.168.3.123:8000/v1`.

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
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B
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
  --prompt "拿起红色杯子"
```

This path skips chassis movement, reports `navigation skipped`, and starts manipulation directly.

## Use Policy-Driven Base Motion

```bash
python -m examples.piper_real.main \
  --use-robot-base \
  --prompt "turn on the water tap." \
  --planner.max-linear-vel 0.25 \
  --planner.max-angular-vel 0.4
```

The runtime still asks for confirmation once before motion. If the policy emits missing, non-finite, or overspeed base commands, the base is stopped and the run aborts.

## Validation

During bring-up, use:

```bash
rostopic echo /cmd_vel
rostopic echo /odom_raw
```

Confirm all of the following:

- Navigation publishes `/cmd_vel` before arm manipulation starts.
- The base returns to zero velocity between planner cycles and on shutdown.
- Policy-driven base commands stay within the configured safety envelope.
- Planner logs show each raw response, rejected or retried cycles, and the terminal reason.
- Navigation failure prevents `Runtime.run()` from starting.
