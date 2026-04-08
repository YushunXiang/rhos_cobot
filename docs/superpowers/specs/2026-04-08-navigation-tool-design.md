# Navigation Tool Refactor

## Overview

Refactor live navigation in `examples/piper_real` from an in-process vision-navigation planner into an explicit tool call.

The new shape is:

`task prompt -> task decomposition -> navigate(prompt) tool / manipulate runtime`

This keeps long-horizon decomposition, but removes `LLMNavigationPlanner` from the live execution path. A `navigate` subtask becomes a high-level function call that executes a fixed TRACER routine.

## Goals

1. Keep `TaskDecomposer` as the component that produces ordered `navigate` and `manipulate` subtasks.
2. Replace live navigation execution with a single reusable module: `examples/piper_real/navigation_tool.py`.
3. Keep `scripts/tracer/tracer_demo.py` as a demo / CLI wrapper only.
4. In v1, ignore prompt semantics during navigation execution: every navigate prompt runs the same fixed routine.
5. Preserve the existing orchestration model in `examples/piper_real/main.py`: navigate failure aborts the task, manipulate subtasks continue to use the current Runtime path.

## Non-Goals

1. No prompt-to-routine routing in v1.
2. No vision-based navigation loop.
3. No new planner model for navigation execution.
4. No change to manipulation policy execution.
5. No change to task decomposition output schema.

## Architecture

### Existing live path

Today `examples/piper_real/main.py` decomposes the full task, then executes navigate subtasks through `examples/piper_real/llm_planner.py` or its fixed-routine helper. Navigation logic is still treated as planner behavior inside the deploy process.

### New live path

The live execution path is reorganized into three clear responsibilities:

1. `examples/piper_real/task_decomposer.py`
Produces ordered `navigate` / `manipulate` subtasks from the top-level prompt. This remains the only LLM-driven planning step in the live navigation path.

2. `examples/piper_real/navigation_tool.py`
Owns execution of a navigate subtask. It receives a natural-language prompt, logs it, executes the fixed TRACER routine, and returns a structured result.

3. `examples/piper_real/main.py`
Owns orchestration only. It decides when to call the navigation tool, when to run manipulation, and when to abort.

`examples/piper_real/llm_planner.py` is removed from the main live path. If the file remains in the repository during migration, it is legacy-only and the refactored live navigation flow does not instantiate or call it.

## Module Design

### New module

Create `examples/piper_real/navigation_tool.py`.

Primary entry point:

```python
def navigate(prompt: str, ros_operator: Any, *, dry_run: bool = False) -> NavigationResult:
    ...
```

Result type:

```python
@dataclasses.dataclass
class NavigationResult:
    ok: bool
    prompt: str
    routine_name: str
    executed_steps: int
    error: str | None = None
```

### v1 behavior

The first version deliberately does not interpret the prompt.

Rules:

1. `prompt` is accepted and logged for observability.
2. Every prompt maps to the same routine: `default_demo`.
3. The routine contents come from the movement sequence currently hard-coded in `scripts/tracer/tracer_demo.py`.
4. The module should expose the routine in reusable Python form instead of keeping the authoritative sequence embedded only in the script.

This keeps the interface aligned with future tool use, while avoiding an unstable routing layer before multiple routines exist.

### Routine execution model

The tool executes a sequence of bounded base commands. Each step contains:

```python
(linear_x, angular_z, duration)
```

Execution behavior:

1. Publish the requested command at a fixed rate for `duration`.
2. Publish a zero command after each step.
3. Publish a zero command again during cleanup or failure handling.

The tool is responsible for the base-motion execution loop. `scripts/tracer/tracer_demo.py` should no longer contain the authoritative motion sequence or the reusable execution logic.

## `tracer_demo.py` Role

`scripts/tracer/tracer_demo.py` remains in the repository as a demo / CLI wrapper only.

Responsibilities:

1. Parse CLI flags such as `--prompt`.
2. Initialize the ROS node and publisher or otherwise acquire the ROS interface needed by `navigation_tool`.
3. Call `examples/piper_real/navigation_tool.py`.
4. Print or log the returned `NavigationResult`.

Non-responsibilities:

1. No embedded business logic for prompt routing.
2. No second copy of the routine sequence.
3. No separate navigation behavior from the reusable module.

## `main.py` Integration

`examples/piper_real/main.py` keeps the current long-horizon structure but changes how navigate subtasks are executed.

### Live orchestration flow

When `use_llm_planner=True` and the top-level prompt is non-empty:

1. Decompose the task into ordered subtasks with `TaskDecomposer`.
2. Log the decomposition result.
3. If any navigate subtask exists and `use_robot_base=True`, perform one safety confirmation before any base motion begins.
4. Create the policy server connection only if at least one manipulate subtask exists and `navigation_only=False`.
5. Create or reuse a `PiperRealEnvironment` when ROS access is needed.
6. For each subtask:
   - `navigate`: call `navigation_tool.navigate(subtask.prompt, ros_operator, dry_run=not args.use_robot_base)`
   - `manipulate`: run the existing Runtime path with `environment.set_prompt(subtask.prompt)`
7. In all exit paths, stop the base if a ROS-backed environment was created.

### Environment ownership

`navigation_tool` should not create `PiperRealEnvironment` itself.

`main.py` remains responsible for environment lifecycle so that:

1. Navigation-only mode can create the minimum environment needed for `ros_operator`.
2. Manipulation mode can share the same environment instance across manipulate subtasks.
3. Cleanup stays centralized.

### Flag semantics

Keep the current flag names for now:

| Flag | Meaning after refactor |
|---|---|
| `use_llm_planner` | Enable task decomposition before execution |
| `use_robot_base` | Allow the navigation tool to publish real base motion |
| `navigation_only` | Execute navigate subtasks only and skip manipulate |

`planner.*` configuration remains relevant for task decomposition only. It is no longer part of live navigation execution.

## Failure Semantics

Navigation failure rules:

1. If the tool cannot execute the fixed routine, it returns `ok=False`.
2. If any navigate subtask returns `ok=False`, `main.py` aborts the entire task immediately.
3. No later manipulate subtasks run after a navigation failure.
4. Base stop is always attempted before returning from the failed path.

Dry-run rules:

1. `dry_run=True` means no real `/cmd_vel` motion is sent.
2. Dry-run still logs the prompt and the routine that would have been executed.
3. Dry-run returns success unless the tool cannot build the routine metadata itself.

## Logging

Use three layers of logging:

1. Subtask-level orchestration log in `main.py`
   - Example: `Executing subtask 1/3 [navigate]: 移动到桌边`
2. Tool-level invocation log in `navigation_tool.py`
   - Example: `Navigation tool invoked: prompt=移动到桌边 routine=default_demo dry_run=False`
3. Step-level execution log in `navigation_tool.py`
   - Example: `Navigation step 2/5: linear_x=0.0 angular_z=0.2 duration=9.0`

This preserves traceability even though v1 ignores prompt semantics during routing.

## Testing

### Unit tests for `navigation_tool.py`

1. Any prompt uses the same `default_demo` routine.
2. The fixed routine steps execute in the expected order.
3. `dry_run=True` avoids real motion publishing.
4. Failure paths return `ok=False` and attempt a stop.

### Unit tests for `scripts/tracer/tracer_demo.py`

1. CLI arguments are parsed correctly.
2. The wrapper calls `navigation_tool.navigate(...)`.
3. The wrapper surfaces the returned result cleanly.

### Unit tests for `examples/piper_real/main.py`

1. Navigate subtasks call the navigation tool instead of `LLMNavigationPlanner`.
2. Navigate failure aborts the remaining subtasks.
3. `navigation_only=True` still skips manipulate subtasks.
4. Manipulate subtasks continue to use the current Runtime path.

### Legacy cleanup

Tests that directly assert live use of `LLMNavigationPlanner` in `main.py` are updated or removed. Any remaining `LLMNavigationPlanner` tests are legacy-only and no longer define expected live behavior.

## File Changes

| File | Change |
|---|---|
| `examples/piper_real/navigation_tool.py` | New reusable navigation tool module |
| `examples/piper_real/main.py` | Replace live navigate execution with tool invocation |
| `scripts/tracer/tracer_demo.py` | Reduce to demo / CLI wrapper around `navigation_tool` |
| `tests/...` | Replace live navigation-planner assumptions with tool-based expectations |
| `docs/...` | Update deploy and usage docs to describe tool-based navigation |

## Migration Notes

1. The live system still decomposes tasks into `navigate` and `manipulate`.
2. The meaning of a navigate subtask changes from "run a planner loop" to "invoke the navigation tool with this prompt."
3. v1 intentionally supports only one fixed base routine for all prompts.
4. Adding multiple routines later should extend `navigation_tool.py` without requiring another orchestration refactor in `main.py`.
