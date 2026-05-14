# Piper Real Deploy Flow

This example runs the real-robot deploy path for `examples/piper_real/main.py`.

The flow is now decomposition-first:

1. Build `PiperRealEnvironment` and connect to the OpenPI policy server only when a manipulate subtask exists.
2. If `--use-llm-planner` is enabled and `--prompt` is non-empty, send the full task prompt to the planner service and get back ordered `navigate` / `manipulate` subtasks.
3. Execute each `navigate` subtask by calling `examples.piper_real.navigation_tool.navigate(...)`, which drives the base through a fixed sequence of body-frame coordinate goals using odometry feedback (same semantics as `scripts/run_tracer_demo_sequence_3term.sh`).
4. Execute each `manipulate` subtask with the existing OpenPI Runtime path.

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

This example expects OpenPI to live next to `rhos_cobot`, not inside
`~/cobot_magic`:

```bash
cd /home/agilex
git clone yushun_github:YushunXiang/OCL-openpi.git openpi
```

The `openpi-client` dependency is resolved as an editable local package from
`../../../openpi/packages/openpi-client`, relative to `examples/piper_real`.

```bash
# Source your ROS environment first.
source /opt/ros/<distro>/setup.bash

# Create or refresh the default uv-managed environment for this example.
cd examples/piper_real
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.10.18

# Run the robot.
UV_PROJECT_ENVIRONMENT=.venv uv run python main.py
```

If you prefer to stay in the repo root, the equivalent command is:

```bash
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --project examples/piper_real --python 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv run --project examples/piper_real python examples/piper_real/main.py
```

`pyproject.toml` is now the source of truth for Python dependencies in this example. The deploy helper defaults to the uv-managed `examples/piper_real/.venv/bin/python`. The legacy `requirements.in` and `requirements.txt` files are kept only for reference. See [`UV_MIGRATION.md`](./UV_MIGRATION.md) for the cross-machine migration workflow.

## Robot Workstation Setup

```bash
sh scripts/init.sh
conda activate aloha
init_deploy
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
UV_PROJECT_ENVIRONMENT=.venv uv run --project examples/piper_real python examples/piper_real/main.py
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

- Accepts text chat-completions requests for task decomposition.
- Accepts replay manipulation replanning requests when `REPLAY_MODE=hybrid`.
- Returns JSON-only planner responses.
- Is reachable from the robot workstation at the configured `--planner.base-url`.

In live deploy and `REPLAY_MODE=hybrid`, the planner does not execute navigation motions. It only returns ordered `navigate` / `manipulate` subtasks; each `navigate` subtask is then executed by the shared local navigation tool.

The planner must return this shape:

```json
{
  "subtasks": [
    {"type": "navigate", "prompt": "move to the table"},
    {"type": "manipulate", "prompt": "pick up the red cup"}
  ]
}
```

## Run Decomposition + Navigation Tool

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
- Each navigate subtask invokes the shared navigation tool.
- In v1, every navigate prompt runs the same fixed `default_demo` routine.
- Manipulation starts only after the navigate subtask returns success.

## Skip Navigation

```bash
python -m examples.piper_real.main \
  --prompt "拿起红色杯子"
```

This path skips chassis movement, reports `navigation skipped`, and starts manipulation directly.

## Dry-Run Navigation

```bash
python -m examples.piper_real.main \
  --use-llm-planner \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://192.168.3.123:8000/v1 \
  --planner.model Qwen/Qwen3.5-4B
```

This runs navigate subtasks through the same tool path but without publishing real base motion. Add `--use-robot-base` to execute the fixed routine on the robot.

## Validation

During bring-up, use:

```bash
rostopic echo /cmd_vel
rostopic echo /odom_raw
```

Confirm all of the following:

- Navigation publishes `/cmd_vel` before arm manipulation starts.
- The base returns to zero velocity between routine steps and on shutdown.
- Task decomposition is logged before subtask execution starts.
- Navigation tool logs show the prompt, selected routine, and each fixed step.
- Navigation failure prevents `Runtime.run()` from starting.
