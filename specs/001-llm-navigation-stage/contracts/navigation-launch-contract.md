# Contract: Navigation Launch and Safety Gate

## Purpose

Define how the existing `examples/piper_real.main` entrypoint enables, configures, and gates the pre-manipulation navigation stage.

## Entry Conditions

Navigation is considered for the current run only when all of the following are true:

- `--prompt` is non-empty.
- `--planner.enable-navigation true` (default).
- Planner connection details (`base_url`, `model`, and usable credentials) are present.

If navigation is disabled, the runtime skips directly to manipulation and reports a `navigation skipped` outcome.

## CLI Surface

The runtime continues to use `tyro` and adds a nested planner config under `Args.planner`.

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--planner.base-url` | string | `http://localhost:8000/v1` | OpenAI-compatible planner endpoint root |
| `--planner.model` | string | `qwen2.5-vl-72b` | Planner model identifier |
| `--planner.api-key` | string | `EMPTY` | Planner API key or placeholder |
| `--planner.max-nav-steps` | int | `20` | Maximum usable planner decisions before navigation fails |
| `--planner.max-linear-vel` | float | `0.3` | Maximum allowed linear x velocity |
| `--planner.max-angular-vel` | float | `0.5` | Maximum allowed angular z velocity |
| `--planner.default-duration` | float | `1.5` | Fallback move duration when an otherwise-usable move omits duration |
| `--planner.enable-navigation` | bool | `true` | Whether to run the pre-manipulation navigation stage |

## Runtime Sequence

1. Construct the current `PiperRealEnvironment`.
2. Obtain `env.ros_operator` from the environment wrapper.
3. If navigation is enabled and a prompt is present, display a safety warning that references the TRACER manual constraints.
4. Require explicit operator confirmation before any chassis movement.
5. If confirmation is granted, run the navigation planner loop.
6. If navigation succeeds, start `Runtime.run()` with the original prompt.
7. If navigation is skipped, start `Runtime.run()` immediately.
8. If navigation fails, do not start manipulation.

## Safety Gate Behavior

- The safety warning must appear before the first planner query that could lead to movement.
- Lack of explicit confirmation is treated as a failed navigation-enabled run, not as an implicit skip.
- No chassis movement may occur before confirmation.
- The base must be commanded to zero velocity before exiting the safety gate path in any terminal state.

## Observability Requirements

The runtime must surface enough information for operators to validate deploy behavior:

- Whether navigation was enabled, skipped, succeeded, or failed.
- Whether operator safety confirmation was received.
- Final navigation termination reason before manipulation handoff or failure.
