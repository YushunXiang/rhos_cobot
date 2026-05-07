# Quickstart: TRACER 2.0 Navigation-First Operation Flow

## 1. Review platform constraints

Before implementation or deploy validation, review the local platform manual:

- `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`

Pay particular attention to the safety sections describing open-area operation and the lack of built-in autonomous obstacle avoidance.

## 2. Prepare the Python environment

```bash
uv venv --python 3.11 examples/piper_real/.venv
source examples/piper_real/.venv/bin/activate
uv pip compile examples/piper_real/requirements.in -o examples/piper_real/requirements.txt --python-version 3.11
uv pip sync examples/piper_real/requirements.txt
```

This quickstart assumes the existing deploy environment already provides the `openpi_client` package required by `examples/piper_real/main.py`.

## 3. Prepare deploy mode on the robot workstation

```bash
sh scripts/init.sh
conda activate aloha
init_deploy
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
source examples/piper_real/.venv/bin/activate
```

If needed, follow the network and ROS setup notes in `docs/deploy.md` before starting the runtime.

## 4. Start the local planner service

Ensure a local OpenAI-compatible multimodal service is reachable from the robot workstation and exposes a chat-completions endpoint at the configured `base_url`.

Minimum expectations:

- Accepts image + text multimodal chat input.
- Can return JSON-only planner decisions.
- Is reachable from the deploy machine over the configured host/port.

## 5. Run the navigation-first deploy flow

```bash
python -m examples.piper_real.main \
  --prompt "移动到桌子旁边拿起红色杯子" \
  --planner.base-url http://localhost:8000/v1 \
  --planner.model qwen2.5-vl-72b
```

Expected runtime flow:

1. The program initializes the existing `PiperRealEnvironment`.
2. If navigation is enabled, it shows a safety warning and asks for explicit operator confirmation.
3. The planner loop issues bounded chassis commands until it returns `stop`, fails, or is skipped.
4. On success or skip, the OpenPI runtime starts arm execution with the original prompt.

## 6. Validate behavior

Use the following checks during bring-up:

```bash
rostopic echo /cmd_vel
```

Validate all of the following:

- Navigation sends `/cmd_vel` commands before manipulation begins.
- Planner logs show each JSON response, any rejected/retried cycle, and the final termination reason.
- The base is commanded to zero velocity between cycles and before handoff.
- Manipulation starts only after navigation success or when navigation is explicitly disabled.

## 7. Bypass navigation when needed

```bash
python -m examples.piper_real.main \
  --prompt "拿起红色杯子" \
  --planner.enable-navigation false
```

This should skip the navigation stage and enter manipulation directly.
