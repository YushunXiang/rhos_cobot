# Implementation Plan: TRACER 2.0 Navigation-First Operation Flow

**Branch**: `[001-llm-navigation-stage]` | **Date**: 2026-03-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-llm-navigation-stage/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Insert an LLM-driven navigation stage into the existing `examples/piper_real` deploy flow before OpenPI manipulation begins. The design keeps all navigation logic inside `examples/piper_real`, reuses the current `RosOperator` camera/odometry subscriptions and `/cmd_vel` publisher, adds a dedicated planner module plus tyro-friendly planner config, enforces TRACER manual safety constraints (explicit operator confirmation, bounded chassis speeds, stop-between-cycles behavior), and preserves the existing prompt and OpenPI runtime handoff once navigation succeeds or is skipped.

## Technical Context

**Language/Version**: Python 3.11 with ROS1 `rospy`  
**Primary Dependencies**: `tyro`, `rospy`, `openpi_client`, `opencv-python`, `openai`, ROS message packages (`nav_msgs`, `geometry_msgs`, `sensor_msgs`), `cv_bridge`  
**Storage**: Local filesystem only for optional logs and generated requirement lockfiles; no new durable datastore  
**Testing**: Manual ROS integration validation on TRACER deploy hardware, plus targeted offline smoke checks for planner-response parsing and command validation logic  
**Target Platform**: Linux robot workstation running ROS1, `examples/piper_real/.venv`, and a locally reachable OpenAI-compatible multimodal planner service  
**Project Type**: Robotics runtime CLI/example inside a Python monorepo  
**Performance Goals**: Keep navigation decisions within configured speed limits (`<=0.3 m/s`, `<=0.5 rad/s` by default), stop the chassis between planner cycles, finish navigation within the configured usable-step budget (default 20) or fail clearly after the 4th consecutive planner query failure, and preserve current OpenPI manipulation runtime behavior after handoff  
**Constraints**: Must not add new ROS subscriptions or nodes for planner sensing; must reuse existing `RosOperator` front camera and odometry buffers; must require explicit operator safety confirmation before movement; must continue using direct `robot_base_publish()` because `use_robot_base` remains `False`; implementation and validation must reference `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`  
**Scale/Scope**: One new planner module and one planner config module, targeted edits to `main.py`, `env.py`, and `requirements.in`/`requirements.txt`, single-robot single-operator deploy flow, no new background services beyond the external planner endpoint

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Initial Gate Review**: PASS WITH CAVEAT. `/home/agilex/rhos_cobot/.specify/memory/constitution.md` still contains placeholder sections rather than project-specific policy, so there are no enforceable constitution gates beyond the feature spec and local safety documentation.

**Applied feature-specific gates for this plan**:
- Reuse existing ROS interfaces rather than introducing duplicate subscriptions or an extra planner node.
- Preserve the current manipulation runtime boundary: navigation completes or is skipped before `Runtime.run()` starts arm execution.
- Encode TRACER manual safety constraints into the runtime design: explicit operator confirmation, bounded velocities, open-area expectation, and no implied autonomous collision avoidance.
- Keep scope limited to the `examples/piper_real` deploy path and its local dependencies.

**Post-design Re-check**: PASS. The planned design stays inside the existing example tree, adds no new durable storage or standalone service, keeps failure handling explicit, and strengthens rather than relaxes documented safety behavior.

## Project Structure

### Documentation (this feature)

```text
specs/001-llm-navigation-stage/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── navigation-launch-contract.md
│   └── planner-response-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
examples/piper_real/
├── main.py
├── env.py
├── real_env.py
├── ros_oper.py
├── requirements.in
├── requirements.txt
├── llm_planner.py          # new: planner client + navigation loop
└── planner_config.py       # new: tyro-friendly planner settings

docs/
└── tracer-2.0-user-manual-v2.0.3-2023.09.pdf

tests/
└── __init__.py
```

**Structure Decision**: Keep all feature-specific implementation in `examples/piper_real` because the current deploy entrypoint, environment wrapper, and ROS integration already live there. Avoid creating a new top-level package or ROS process; the only code-level surface added is a planner helper plus configuration dataclass consumed by the existing CLI entrypoint.

## Complexity Tracking

No constitution violations requiring justification were identified.
