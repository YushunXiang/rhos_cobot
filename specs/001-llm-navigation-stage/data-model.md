# Data Model: TRACER 2.0 Navigation-First Operation Flow

## Overview

This feature adds transient in-memory navigation state around the existing `examples/piper_real` deploy runtime. No new durable database or persisted domain store is introduced. The model centers on a single navigation session that runs before manipulation and produces a handoff outcome.

## Entities

### Task Prompt

- **Purpose**: Represents the operator's end-to-end instruction for the deploy run.
- **Fields**:
  - `prompt_text`: original natural-language task string.
  - `navigation_enabled`: whether the pre-manipulation navigation stage is active for this run.
  - `safety_confirmed`: whether the operator explicitly approved chassis movement for this run.
  - `manipulation_prompt_text`: prompt handed unchanged to the OpenPI runtime after navigation success or skip.
- **Validation rules**:
  - `prompt_text` must be non-empty for navigation-enabled runs.
  - `manipulation_prompt_text` must remain identical to `prompt_text`.
- **Relationships**:
  - One task prompt creates at most one navigation session per deploy run.

### Planner Settings

- **Purpose**: Operator-supplied runtime configuration for the planner integration.
- **Fields**:
  - `base_url`: OpenAI-compatible planner endpoint root.
  - `model`: planner model identifier.
  - `api_key`: planner authentication token or placeholder value.
  - `max_nav_steps`: maximum number of usable planner decisions allowed in one navigation session.
  - `max_linear_vel`: maximum allowed linear x velocity.
  - `max_angular_vel`: maximum allowed angular z velocity.
  - `default_duration`: fallback move duration when a usable move response omits duration.
  - `enable_navigation`: top-level switch for bypassing navigation.
- **Validation rules**:
  - `max_nav_steps` must be positive.
  - `max_linear_vel`, `max_angular_vel`, and `default_duration` must be positive.
  - `base_url` and `model` must be set when navigation is enabled.

### Scene and Position Snapshot

- **Purpose**: The latest sensing context sent to the planner each cycle.
- **Fields**:
  - `front_image_b64`: most recent front-camera frame encoded as a data URI.
  - `odom_x`: current base x position.
  - `odom_y`: current base y position.
  - `odom_yaw`: current base heading in radians.
  - `history_summary`: compact summary of previous planner decisions and execution outcomes.
- **Validation rules**:
  - A navigation cycle requires both a current image and a current odometry sample.
  - Missing image or odometry marks the cycle as an immediate navigation failure.

### Planner Decision

- **Purpose**: The normalized control intent returned by the external planner.
- **Fields**:
  - `action`: `move` or `stop`.
  - `linear_x`: requested linear velocity for `move`.
  - `angular_z`: requested angular velocity for `move`.
  - `duration`: requested move duration or fallback duration.
  - `reasoning`: free-form planner explanation for a move.
  - `reason`: free-form planner explanation for a stop.
  - `raw_payload`: raw JSON/body captured for logging and troubleshooting.
  - `is_valid`: whether the response passed schema and safety validation.
- **Validation rules**:
  - `action` must be exactly `move` or `stop`.
  - `move` decisions require bounded `linear_x` and `angular_z` values.
  - Overspeed `move` decisions are rejected rather than clamped.
  - Invalid or unavailable planner responses increment the consecutive planner failure counter.

### Navigation Session

- **Purpose**: Tracks one complete pre-manipulation navigation attempt.
- **Fields**:
  - `status`: `awaiting_confirmation`, `running`, `succeeded`, `failed`, or `skipped`.
  - `usable_steps_consumed`: count of usable planner decisions processed.
  - `usable_step_limit`: copied from planner settings.
  - `consecutive_planner_failures`: current count of back-to-back invalid/unavailable planner responses.
  - `planner_failure_limit`: fixed at 3 retries, with failure on the 4th consecutive failure.
  - `last_decision`: most recent normalized planner decision.
  - `last_executed_velocity`: last velocity pair sent to `/cmd_vel`.
  - `termination_reason`: final success/skip/failure reason.
- **Validation rules**:
  - Planner failures do not increment `usable_steps_consumed`.
  - The session must end with the base stopped before success, failure, or handoff.

### Handoff Outcome

- **Purpose**: Determines whether manipulation may start after navigation handling.
- **Fields**:
  - `result`: `success`, `skip`, or `failure`.
  - `can_start_manipulation`: boolean derived from result.
  - `reason`: operator-facing explanation of the terminal state.
- **Validation rules**:
  - `success` and `skip` allow manipulation.
  - `failure` blocks manipulation.

## State Transitions

### Navigation Session

```text
awaiting_confirmation -> running
awaiting_confirmation -> failed        # operator declines or does not confirm safety
running -> running                     # another cycle executes
running -> succeeded                   # planner returns stop and base is stationary
running -> failed                      # missing data, invalid response threshold exceeded, step limit reached, or other terminal error
awaiting_confirmation -> skipped       # navigation disabled before session starts
```

### Planner Decision Lifecycle

```text
raw response -> parsed -> validated -> executable move
raw response -> parsed -> validated -> stop
raw response -> parsed -> rejected     # schema error, overspeed, unsupported action, nonrecoverable invalid payload
raw response -> unavailable            # transport/API failure
```

## Derived Rules

- A navigation-enabled run must preserve the original task prompt for the later manipulation stage.
- The base is stopped while waiting for planner output, after each move cycle, and before handoff/failure.
- Safety confirmation is required only when navigation is enabled.
- All implementation and validation decisions must stay consistent with `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`.
