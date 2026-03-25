# Replay Pure-Offline Mode & Two-Layer LLM Planner Architecture

## Overview

Two independent changes to `examples/piper_real/main.py` and surrounding modules:

1. **Replay pure-offline mode**: `--replay-dataset` reads HDF5 ground-truth actions and prints them without connecting to an inference server.
2. **Two-layer LLM planner**: `--use-llm-planner` decomposes long-horizon tasks into ordered subtask lists (navigate + manipulate), then executes them sequentially.

## Requirement 1: Replay Pure-Offline Mode

### Current behavior

When `--replay-dataset` is set, `main.py` creates a `WebsocketClientPolicy`, connects to the inference server, runs the `Runtime` loop feeding HDF5 observations to the server, and compares predicted actions with ground-truth.

### New behavior

When `--replay-dataset` is set, `main.py` loads the HDF5 file and iterates over ground-truth actions, printing each step. No inference server connection, no `Runtime`, no ROS.

**Note**: This intentionally removes the inference-comparison capability from replay mode. The previous behavior (connect to server, predict, compute MAE) is no longer needed — replay mode is now a pure data inspection tool.

### Flow

```
replay_dataset is set
  -> load HDF5 (qpos, action, base_action if present)
  -> for each step:
       log: "Replay step {i}/{total} -- arm: [...14 floats...] base: [lin_x, ang_z] or N/A"
  -> log: "Replay complete: {total} steps, action_dim={dim}, has_base_action={bool}"
  -> return
```

### Output format

Per-step structured log line with full precision (`%.4f`), all 14 arm values printed:
```
Replay step 0/499 -- arm: [0.0010, -0.0030, 0.0158, ...all 14...] base: [0.2000, -0.1000]
```

If HDF5 has no `base_action` key or action dimension < 16, base shows `N/A`.

Summary at end:
```
Replay complete: 500 steps, action_dim=14, has_base_action=True
```

### Ignored parameters

`--host` and `--port` are silently ignored in replay mode (no error, just unused).

### Files changed

- `main.py`: replace the replay branch (lines 50-91) with direct HDF5 iteration and print loop. Remove `WebsocketClientPolicy`, `ActionChunkBroker`, `PolicyAgent`, `Runtime` creation from this branch.

## Requirement 2: Two-Layer LLM Planner Architecture

### Architecture

Two layers:

1. **TaskDecomposer** (new): one-shot LLM call to decompose a full prompt into an ordered list of subtasks.
2. **LLMNavigationPlanner** (existing): executes a single navigate subtask via its multi-step navigation loop.

### New file: `examples/piper_real/task_decomposer.py`

```python
@dataclasses.dataclass
class Subtask:
    type: str    # "navigate" | "manipulate"
    prompt: str  # description for this subtask

class TaskDecomposer:
    def __init__(self, config: PlannerConfig): ...
    def decompose(self, task_prompt: str) -> list[Subtask]: ...
```

#### LLM interaction

System prompt instructs the LLM to return JSON only:
```json
{
  "subtasks": [
    {"type": "navigate", "prompt": "move to kitchen table"},
    {"type": "manipulate", "prompt": "pick up the red cup"},
    {"type": "navigate", "prompt": "move to the living room coffee table"},
    {"type": "manipulate", "prompt": "place the cup on the table"}
  ]
}
```

Validation:
- `subtasks` must be a non-empty list.
- Each entry must have `type` in `{"navigate", "manipulate"}` and a non-empty `prompt`.
- On invalid response, retry up to 3 times (4 attempts total, no delay between retries). On final failure, raise an exception that propagates to `main.py`, which logs the error and exits.
- Maximum subtask count: 10. If the LLM returns more than 10 subtasks, treat as invalid response and retry.

#### Single-manipulate prompts

If the prompt describes a pure manipulation task with no navigation (e.g., "pick up the cup"), the decomposer should return a single-element list: `[{"type": "manipulate", "prompt": "pick up the cup"}]`. The system prompt for the decomposer must explicitly encourage this.

#### `set_prompt()` method

`PiperRealEnvironment.set_prompt(prompt: str)` simply sets `self._prompt = prompt`. It does NOT propagate to the underlying `_real_env` or `ros_operator` — the prompt is only consumed in `get_observation()` return dict.

#### Config reuse

`TaskDecomposer` reuses `PlannerConfig` (same `base_url`, `model`, `api_key`).

### LLMNavigationPlanner changes

No structural changes. `run(task_prompt)` now receives a single navigate subtask prompt (e.g., "move to kitchen") instead of the full task description. This is a calling convention change only.

### main.py subtask execution loop

When `use_llm_planner=True` and prompt is non-empty (if prompt is empty, skip decomposition and log "LLM planner enabled but prompt is empty; skipping to stationary manipulation"):

```
1. TaskDecomposer.decompose(prompt) -> subtask_list
2. Log decomposition result
3. Safety confirmation (once, if use_robot_base=True)
4. For each subtask in subtask_list:
     if subtask.type == "navigate":
       if use_robot_base:
         planner.run(subtask.prompt)  # actual base movement
         if failed: stop_base, abort entire task
       else:
         log "Navigate (dry-run): {subtask.prompt}"  # print only
     elif subtask.type == "manipulate":
       if navigation_only:
         log "Manipulate (skipped): {subtask.prompt}"
         continue
       create Runtime with prompt=subtask.prompt, run()
5. Cleanup: stop_base if use_robot_base
```

### Key behaviors

- **Each manipulate subtask starts an independent Runtime run** with that subtask's prompt.
- **Navigate failure aborts the entire task** -- stop_base, no further subtasks.
- **Safety confirmation**: once before the first subtask that requires base motion (same as current).
- **WebsocketClientPolicy creation**: created once upfront before the subtask loop, but only if the subtask list contains at least one manipulate entry AND `navigation_only` is false. Otherwise no server connection.
- **PiperRealEnvironment lifecycle**: created once upfront. Between manipulate subtasks, `reset()` is called. The prompt is updated per subtask via a new `set_prompt(str)` method on `PiperRealEnvironment`.
- **Action truncation**: `env.py` `apply_action` truncates the action array to 14 dims BEFORE passing to `real_env.step()`. `real_env.step()` continues to assume 14-dim input with 7/7 split (left arm + right arm). This is a safety-critical boundary — truncation must happen in `env.py`, not `real_env.py`.
- **`ros_config["use_robot_base"]`**: hardcoded to `False` in `real_env.py`; the base publish block in `step()` (lines 252-254) and `_extract_base_velocity` are removed entirely. Base movement only happens via `LLMNavigationPlanner`.

### Flag semantics

| Flag | New semantics |
|---|---|
| `use_llm_planner` | Enable two-layer architecture: decompose then execute |
| `use_robot_base` | Whether navigate subtasks actually move the base |
| `navigation_only` | Only execute navigate subtasks, skip manipulate |

### Flag combination matrix

| `use_llm_planner` | `use_robot_base` | `navigation_only` | Behavior |
|---|---|---|---|
| `False` | `False` | `False` | Stationary manipulation (current default) |
| `False` | `True` | `False` | **Validation error**: `use_robot_base` requires `use_llm_planner` in the new architecture |
| `False` | `*` | `True` | **Validation error**: `navigation_only` requires `use_llm_planner` |
| `True` | `True` | `False` | Decompose -> navigate (move) + manipulate (execute) |
| `True` | `False` | `False` | Decompose -> navigate (dry-run print) + manipulate (execute) |
| `True` | `True` | `True` | Decompose -> navigate (move) only, skip manipulate |
| `True` | `False` | `True` | Decompose -> navigate (dry-run print) only, skip manipulate |
| `*` | `*` | replay | Pure HDF5 playback, print actions, no server |

### Validation rules

- `--replay-dataset` and `--navigation-only` are mutually exclusive (existing check).
- `--replay-dataset` and `--use-llm-planner` are mutually exclusive (**new check** to add).
- `--use-robot-base` requires `--use-llm-planner` (**new check**; the old mode of policy-driven base velocity during manipulation is removed).
- `--navigation-only` requires `--use-llm-planner` (**new explicit check**; current code implicitly coerces this, new code should validate and error).

All validation checks go at the top of `main()`, before any resource creation, matching the existing pattern (lines 42-47).

## Files changed summary

| File | Change |
|---|---|
| `examples/piper_real/main.py` | Replay branch rewrite; new subtask execution loop |
| `examples/piper_real/task_decomposer.py` | New file: TaskDecomposer + Subtask dataclass |
| `examples/piper_real/env.py` | Remove `use_robot_base` / base velocity params; add `set_prompt()` method; truncate action to 14 dims in `apply_action` |
| `examples/piper_real/real_env.py` | Remove `_extract_base_velocity` from `step()`, remove base publish in step; remove `use_robot_base` param |
| `examples/piper_real/planner_config.py` | No changes needed |
| `examples/piper_real/base_safety.py` | No changes needed |
| `examples/piper_real/llm_planner.py` | Remove `_extract_json_text` (moved to `llm_utils.py`), import from shared utility |
| `examples/piper_real/llm_utils.py` | New file: shared `extract_json_text` utility |
| `docs/deploy.md` | Update CLI docs, flag semantics, examples, new validation rules |

## Implementation notes

### Decomposition logging

After `TaskDecomposer.decompose()`, log the full subtask list in JSON format:
```
Task decomposition: {"subtasks": [{"type": "navigate", "prompt": "..."}, ...], "count": 4}
```

### Shared JSON extraction

`LLMNavigationPlanner._extract_json_text` should be extracted into a shared utility (e.g., `llm_utils.py`) since `TaskDecomposer` needs identical JSON-from-LLM-response parsing.

### Runtime per manipulate subtask

Each manipulate subtask uses the same `Runtime` configuration (`max_hz=50`, `num_episodes=1`, `max_episode_steps` from args). The environment is reused but `reset()` is called and prompt is updated via `set_prompt()` before each run.
