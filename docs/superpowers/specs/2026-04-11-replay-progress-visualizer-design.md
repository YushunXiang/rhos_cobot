# Replay Progress Visualizer Design

**Date:** 2026-04-11  
**Scope:** Offline replay-only visual debugger for PI0/PI0.5 manipulation progress

## Goal

Build an offline visualization tool that replays a single HDF5 manipulation segment, runs a local progress-head checkpoint step-by-step, applies the same progress-first decision logic used by the replay/hybrid path, optionally falls back to the VLM replanner, and renders the result as an MP4 with an optional JSONL trace.

This tool is for debugging and inspection, not for live control.

## Non-Goals

- No live robot integration
- No websocket/server-based policy inference
- No interactive Web UI in this iteration
- No navigate-subtask visualization in this iteration
- No implementation of full long-horizon task decomposition in this iteration

## User-Facing Behavior

The new script will live at:

- `rhos_cobot/scripts/visualize_replay_progress.py`

It will accept at least:

- `--dataset-path`
- `--checkpoint-dir`
- `--output-video`
- `--prompt`
- `--start-step`
- `--end-step`

Optional arguments:

- `--dump-jsonl`
- `--camera-name`
- `--task-decompose`

`--task-decompose` is a reserved extension point in this iteration. If provided, the script should fail clearly with a not-yet-implemented message rather than silently ignoring it.

Default mode is a single manipulate prompt over a specified replay range.

## Inputs

### Replay Data

The script uses `ReplayEnvironment` from `examples/piper_real/replay_env.py` to read the HDF5 episode and fetch image/state observations. Camera frames are read through the existing replay environment APIs so the visualization stays aligned with the current replay stack.

### Policy Checkpoint

The script loads the checkpoint locally through `openpi.policies.policy_config.create_trained_policy(...)`.

The tool requires a checkpoint with a trained progress head:

- if `progress_metadata.json` is missing, or
- if `has_progress_head` is false,

then the script should stop with a clear error explaining that this visualizer requires a progress-head checkpoint.

### Prompt and Range

The user provides a single manipulate prompt and a replay step range. The script treats that range as the active manipulation segment to visualize.

## Output

### Primary Output

An MP4 video file showing:

- replay camera frame
- current step index
- current prompt
- current progress value
- progress curve over time
- complete threshold reference line
- event markers for:
  - progress complete
  - progress stall
  - progress regression
  - replanner continue
  - replanner complete
- trigger reason text
- replanner decision text when called

### Secondary Output

Optional JSONL output when `--dump-jsonl` is set.

Each line records one replay step with at least:

- `step`
- `prompt`
- `progress`
- `progress_event`
- `trigger_reason`
- `replanner_called`
- `replanner_action`
- `replanner_reason`
- `completed`
- `camera_name`

## Recommended Architecture

Use a single script with internally separated components:

### 1. ReplayProgressRunner

Responsibilities:

- drive replay step iteration
- load local policy
- call policy inference each step
- read `progress`
- apply replay/hybrid-compatible progress monitoring
- call replanner only on fallback paths
- emit structured per-step records

The runner should reuse the same decision semantics already introduced in `examples/piper_real/main.py`:

- complete when `progress >= complete_threshold`
- stall when the recent progress window moves less than `stall_threshold`
- regression when current progress falls behind previous max by `regression_threshold`

### 2. ReplayDecisionRecorder

Responsibilities:

- store per-step structured records in memory
- optionally write JSONL records incrementally

This creates a stable intermediate representation that later extensions can reuse.

### 3. ReplayFrameComposer

Responsibilities:

- fetch the target camera frame for the current replay step
- render overlays using OpenCV
- draw progress curve, threshold line, text labels, and event markers

### 4. Video Output

Responsibilities:

- initialize `cv2.VideoWriter`
- write composed frames
- close cleanly on completion or error

## Why This Shape

Three candidate shapes were considered:

### Option A: Single-pass script with internal helpers

Pros:

- fastest to implement
- minimal moving parts
- easiest to keep aligned with current replay logic

Cons:

- script can grow if helper boundaries are not maintained

### Option B: Two-stage pipeline (trace first, render second)

Pros:

- strongest separation between inference and rendering
- easiest to extend later

Cons:

- heavier than needed for the current request

### Option C: JSON-only tool first

Pros:

- simplest implementation

Cons:

- does not satisfy the primary MP4 visualization goal

Recommended choice: Option A, implemented internally using the decomposition style of Option B.

## Rendering Layout

The MP4 should use a fixed layout with three information zones:

### Main Camera Pane

Shows the selected replay camera frame for the current step.

### Progress Plot Pane

Shows:

- progress history polyline
- horizontal `complete_threshold` reference line
- markers for complete/stall/regression events
- markers for replanner outcomes when replanner is called

### Text / Status Pane

Shows:

- current prompt
- current step
- current progress
- current event
- current trigger reason
- replanner action and reason if available

The renderer should prioritize readability over styling complexity.

## Policy Inference Path

The visualization tool should load the policy locally instead of using:

- websocket server
- `ActionChunkBroker`
- `PolicyAgent`

Reason:

- this is an offline debugging tool
- the local path removes unnecessary runtime/network dependencies
- the checkpoint directory is already available as a direct input

## Replanner Cooperation

The tool should follow the same high-level contract as replay/hybrid:

- progress-first for normal flow
- replanner as fallback on stall/regression
- optional final confirmation after progress-complete can be added later, but is not required in this initial tool unless already exposed by config

If replanner invocation fails:

- continue generating the visualization
- mark the step record with a replanner error
- annotate the frame accordingly

## Extension Hook for `--task-decompose`

The script should reserve a branch for future multi-subtask operation:

- parse `--task-decompose`
- define a subtask-spec abstraction internally
- for now, raise `NotImplementedError` with a clear message

The current single-prompt flow should already be structured so a future decomposed task can produce:

- multiple manipulate segments
- one combined video or multiple per-segment outputs

## Error Handling

The script should fail early and clearly for:

- missing dataset file
- invalid checkpoint directory
- progress-head metadata missing or false
- invalid replay range
- unknown camera name

The script should not crash the whole run for:

- replanner runtime/API errors after replay starts

Instead, it should annotate the output and continue if possible.

## Testing Plan

Add tests for:

### CLI / Config

- required argument validation
- range validation
- `--task-decompose` currently raises a clear not-implemented error

### Runner Logic

- progress records captured each step
- stall/regression/complete events mapped correctly
- replanner is called only on fallback triggers

### Renderer

- composed frame has expected output size
- overlay drawing succeeds on synthetic input

### Lightweight Integration

- fake replay + fake policy + fake replanner produces:
  - step records
  - JSONL output when requested
  - frames written to video writer

## Risks

### Risk: Script grows too large

Mitigation:

- keep runner, recorder, and renderer as separate classes/functions inside the script

### Risk: Replay semantics drift from `main.py`

Mitigation:

- reuse `ReplayTaskProgressTracker`
- keep threshold parameters sourced from the same planner config conventions

### Risk: Rendering dependencies become fragile

Mitigation:

- prefer OpenCV-only rendering instead of matplotlib in the frame loop

## Open Questions Resolved

- Default flow is single manipulate prompt, not long-horizon decomposition
- Main output is MP4, with optional JSONL
- `--task-decompose` is an extension hook only in this iteration
