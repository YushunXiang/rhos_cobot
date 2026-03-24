# Contract: Planner Request and Response

## Purpose

Define the external planner interaction used by the navigation stage.

## Planner Request Shape

The navigation stage sends one multimodal chat-completions request per cycle. Each request includes:

- A system instruction requiring JSON-only output.
- The operator's task prompt.
- A compact odometry summary `{x, y, yaw}`.
- Recent navigation history (prior decisions, rejections, retries, and terminal progress hints).
- The latest front-camera frame as a base64 data URI image.

## Allowed Response Shapes

### Move

```json
{
  "action": "move",
  "linear_x": 0.2,
  "angular_z": -0.1,
  "duration": 1.2,
  "reasoning": "Rotate slightly right and advance toward the table edge."
}
```

### Stop

```json
{
  "action": "stop",
  "reason": "The robot is already in a usable operating position."
}
```

## Validation Rules

- Only `move` and `stop` actions are accepted.
- `move.linear_x` must be within `[-max_linear_vel, max_linear_vel]`.
- `move.angular_z` must be within `[-max_angular_vel, max_angular_vel]`.
- Overspeed commands are rejected and do not execute.
- If `move.duration` is missing, the runtime uses `PlannerConfig.default_duration`.
- If the planner response is malformed, unsupported, or unavailable, the cycle counts as a planner failure.
- Planner failures retry up to 3 consecutive times and fail navigation on the 4th consecutive failure.
- Planner failures do not consume the navigation step budget.

## Execution Semantics

- The base remains stopped while the runtime waits for planner output.
- Each accepted `move` executes as one bounded cycle and is followed immediately by a zero-velocity command.
- A validated `stop` response ends navigation successfully only after the runtime confirms the base has been commanded to remain stationary.
- Any terminal failure path commands zero velocity before navigation exits.

## Logging Contract

Each cycle log record should capture:

- Cycle index and usable step count.
- Current odometry summary.
- Raw planner payload or transport error.
- Normalized command outcome (`move`, `stop`, `rejected`, `retrying`, or `failed`).
- Executed velocity pair and duration when movement occurs.
- Terminal reason when the session ends.
