# Tasks: TRACER 2.0 Navigation-First Operation Flow

**Input**: Design documents from `/specs/001-llm-navigation-stage/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: No new automated test tasks are generated. Verification for this feature follows the manual ROS/deploy validation flow defined in the spec and quickstart.

**Organization**: Tasks are grouped by user story to enable independent implementation and validation of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g. US1, US2, US3)
- Every task includes the exact file path to change

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare local dependencies and create the new planner files needed by the feature.

- [X] T001 Add the `openai` dependency to `examples/piper_real/requirements.in`
- [X] T002 [P] Create the planner settings dataclass scaffold in `examples/piper_real/planner_config.py`
- [X] T003 [P] Create the navigation planner module scaffold in `examples/piper_real/llm_planner.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared runtime hooks that all user stories depend on.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T004 [P] Expose the underlying ROS operator through a public property in `examples/piper_real/env.py`
- [X] T005 Extend the CLI argument model to include planner settings in `examples/piper_real/main.py`
- [X] T006 Implement shared image encoding, odometry extraction, yaw conversion, and base-stop helpers in `examples/piper_real/llm_planner.py`
- [X] T007 Define common navigation status/logging helpers used by all terminal outcomes in `examples/piper_real/llm_planner.py`

**Checkpoint**: Foundation ready. The planner can now be wired into the deploy flow story by story.

---

## Phase 3: User Story 1 - Reach the work area before manipulation (Priority: P1) 🎯 MVP

**Goal**: Run a navigation stage before manipulation and hand off to OpenPI only after the planner determines the base is in a usable operating position.

**Independent Test**: Start a navigation-enabled run with a prompt that requires chassis repositioning. The system should query the planner, publish bounded `/cmd_vel` commands, stop between cycles, and start manipulation only after navigation succeeds.

### Implementation for User Story 1

- [X] T008 [US1] Implement the OpenAI-compatible multimodal request builder and strict JSON response parsing in `examples/piper_real/llm_planner.py`
- [X] T009 [US1] Implement the navigation success loop, usable-step accounting, and stop-between-cycles motion execution in `examples/piper_real/llm_planner.py`
- [X] T010 [US1] Invoke `LLMNavigationPlanner.run()` before `Runtime.run()` in `examples/piper_real/main.py`
- [X] T011 [US1] Preserve the original task prompt across navigation handoff in `examples/piper_real/main.py` and `examples/piper_real/llm_planner.py`

**Checkpoint**: User Story 1 is complete when a navigation-enabled run moves first, then starts manipulation only after successful arrival.

---

## Phase 4: User Story 2 - Bypass navigation when it is unnecessary (Priority: P2)

**Goal**: Allow operators to disable navigation and go directly into manipulation with the same entrypoint and prompt flow.

**Independent Test**: Start the runtime with `--planner.enable-navigation false`. The system should skip planner execution, publish no pre-operation base movement, report `navigation skipped`, and begin manipulation immediately.

### Implementation for User Story 2

- [X] T012 [US2] Implement the `enable_navigation` bypass gate and prompt guard in `examples/piper_real/main.py`
- [X] T013 [P] [US2] Apply operator-provided planner connection details, step limits, and motion defaults from `examples/piper_real/planner_config.py` inside `examples/piper_real/llm_planner.py`
- [X] T014 [US2] Surface explicit `navigation skipped` status and direct manipulation handoff messaging in `examples/piper_real/main.py` and `examples/piper_real/llm_planner.py`

**Checkpoint**: User Story 2 is complete when operators can disable navigation cleanly without affecting the manipulation-only path.

---

## Phase 5: User Story 3 - Stop safely and explain why navigation did not complete (Priority: P3)

**Goal**: Enforce safety confirmation and robust failure handling so navigation failures stop the base, block manipulation, and leave operators with actionable diagnostics.

**Independent Test**: Trigger failure paths such as missing operator confirmation, overspeed planner output, repeated invalid planner responses, or missing odometry/image data. The system should keep the base stopped, log the failure reason, and refuse to start manipulation.

### Implementation for User Story 3

- [X] T015 [US3] Add the operator safety confirmation gate before any chassis movement in `examples/piper_real/main.py` and `examples/piper_real/llm_planner.py`
- [X] T016 [US3] Enforce overspeed rejection, 3-strike planner retry limits, and non-counting planner failures in `examples/piper_real/llm_planner.py`
- [X] T017 [US3] Guarantee zero-velocity cleanup on every terminal path and block `Runtime.run()` after navigation failure in `examples/piper_real/main.py` and `examples/piper_real/llm_planner.py`
- [X] T018 [P] [US3] Log raw planner payloads, normalized decisions, executed velocities, and terminal reasons in `examples/piper_real/llm_planner.py`

**Checkpoint**: User Story 3 is complete when all configured failure cases produce a stopped base, a blocked manipulation stage, and clear operator-facing diagnostics.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finalize dependency lockfiles and operator-facing documentation.

- [X] T019 Regenerate `examples/piper_real/requirements.txt` from `examples/piper_real/requirements.in`
- [X] T020 [P] Update operator usage and planner/manual references in `examples/piper_real/README.md` and `docs/deploy.md`
- [X] T021 Validate the final operator workflow against `specs/001-llm-navigation-stage/quickstart.md` and align any remaining command examples in `examples/piper_real/README.md` and `docs/deploy.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: No dependencies. Start immediately.
- **Phase 2: Foundational**: Depends on Setup completion. Blocks all user stories.
- **Phase 3: User Story 1 (P1)**: Depends on Foundational completion.
- **Phase 4: User Story 2 (P2)**: Depends on Foundational completion and should be applied after US1 because it refines the same deploy entrypoint and planner configuration path.
- **Phase 5: User Story 3 (P3)**: Depends on Foundational completion and should be applied after US1 because it extends the navigation loop and terminal control behavior.
- **Phase 6: Polish**: Depends on the desired user stories being complete.

### User Story Dependencies

- **US1**: No dependency on other user stories after Foundational.
- **US2**: Reuses the planner configuration and entrypoint flow established for US1, but remains independently verifiable through the navigation-disabled path.
- **US3**: Reuses the navigation loop introduced for US1, but remains independently verifiable through failure-path scenarios.

### Within Each User Story

- Shared config and runtime hooks before orchestration changes.
- Planner request/response handling before execution/handoff logic.
- Failure validation and cleanup before documentation polish.

---

## Parallel Opportunities

- **Setup**: `T002` and `T003` can run in parallel because they create different new files.
- **Foundational**: `T004` can run in parallel with `T006` because they touch different files and both feed later wiring.
- **US2**: `T013` can run in parallel with `T012` after the foundational planner config exists.
- **US3**: `T018` can run in parallel with `T016` because logging enrichment can be added alongside failure-rule enforcement in the same story window after the planner loop exists.
- **Polish**: `T020` can run in parallel with `T019` because docs and lockfile updates touch different files.

---

## Parallel Example: User Story 2

```bash
# Once User Story 1 is merged and planner config exists:
Task: "Implement the enable_navigation bypass gate and prompt guard in examples/piper_real/main.py"
Task: "Apply operator-provided planner connection details, step limits, and motion defaults from examples/piper_real/planner_config.py inside examples/piper_real/llm_planner.py"
```

---

## Parallel Example: User Story 3

```bash
# After the navigation loop is working:
Task: "Enforce overspeed rejection, 3-strike planner retry limits, and non-counting planner failures in examples/piper_real/llm_planner.py"
Task: "Log raw planner payloads, normalized decisions, executed velocities, and terminal reasons in examples/piper_real/llm_planner.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Stop and validate the navigation-first handoff using `/cmd_vel` and planner logs.
5. Demo or deploy the MVP once navigation-first execution is stable.

### Incremental Delivery

1. Build the planner scaffolding and shared runtime hooks.
2. Deliver US1 so the robot can navigate before manipulation.
3. Add US2 so operators can intentionally bypass navigation without breaking the entrypoint.
4. Add US3 so failure handling, retries, and safety confirmation are enforced.
5. Finish with dependency lockfile refresh and operator documentation updates.

### Suggested MVP Scope

- **MVP**: Phase 1 + Phase 2 + Phase 3 (User Story 1 only)
- **Second increment**: Phase 4 (User Story 2)
- **Third increment**: Phase 5 (User Story 3)
- **Final hardening**: Phase 6 (Polish)

---

## Notes

- Every task follows the required checklist format: checkbox, ID, optional `[P]`, optional `[US#]`, and exact file paths.
- Automated tests were not added because the feature specification defined deploy-mode validation rather than a TDD requirement.
- Use `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf` during implementation and validation.
