# Feature Specification: TRACER 2.0 Navigation-First Operation Flow

**Feature Branch**: `[001-llm-navigation-stage]`  
**Created**: 2026-03-24  
**Status**: Draft  
**Input**: User description: "为 TRACER 2.0 增加先导航后操作的工作流，使用外接视觉规划器根据当前场景和底盘状态决定移动，达到目标位置后再开始手臂操作"

## Clarifications

### Session 2026-03-24

- Q: Should navigation require an explicit operator safety confirmation before it starts? → A: Yes. When navigation is enabled, the operator must explicitly confirm safety before navigation may begin.

- Q: How should the system handle planner commands that exceed configured speed limits? → A: Reject that cycle's command, keep the base stopped, and request a new planner decision on the next cycle.
- Q: How many consecutive planner query failures or invalid planner responses should be retried before navigation fails? → A: Retry up to 3 consecutive failures while keeping the base stopped; fail navigation on the 4th consecutive failure.
- Q: Should planner query failures count toward the maximum navigation step limit? → A: No. Only usable planner decisions count toward the navigation step limit; planner failures count only toward the consecutive planner failure limit.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reach the work area before manipulation (Priority: P1)

作为部署操作员，我可以用同一个任务提示词启动整条流程，让机器人先移动到适合执行操作的位置，再开始手臂任务。

**Why this priority**: 如果机器人没有先到达合适站位，后续操作阶段即使启动也很可能失败，因此这是整条流程的前提。

**Independent Test**: 在启用导航的前提下启动一个需要移动到底盘目标位置的任务，验证系统先进入导航阶段，只有在确认到位后才开始操作阶段。

**Acceptance Scenarios**:

1. **Given** navigation is enabled, a task prompt is provided, and the planner is available, **When** the operator explicitly confirms the navigation safety warning and starts a task that requires the robot to approach a workstation, **Then** the system enters a navigation stage before manipulation begins.
2. **Given** the navigation stage determines that the robot has reached a usable operating position, **When** navigation ends, **Then** the base is stationary and the manipulation stage starts with the same task objective.
3. **Given** the robot is already in a suitable position at task start, **When** the planner indicates arrival immediately, **Then** the system skips further movement and hands off directly to manipulation.

---

### User Story 2 - Bypass navigation when it is unnecessary (Priority: P2)

作为部署操作员，我可以在启动任务时关闭导航，让系统直接进入操作阶段，用于机器人已经就位的场景。

**Why this priority**: 不是所有任务都需要移动到底盘目标位置，跳过导航可以减少准备时间并保留现有操作流程。

**Independent Test**: 在关闭导航的情况下启动任务，验证系统不会进入导航阶段，也不会发送底盘移动指令，而是直接开始操作阶段。

**Acceptance Scenarios**:

1. **Given** navigation is disabled at task start, **When** the operator launches a task, **Then** the system bypasses the navigation stage and begins manipulation immediately.
2. **Given** navigation is disabled, **When** the task runs, **Then** the system reports that navigation was skipped and does not issue any base movement for the pre-operation stage.
3. **Given** the operator provides startup navigation settings, **When** a job begins, **Then** the system applies those settings to the current run without requiring code changes or a separate setup workflow.

---

### User Story 3 - Stop safely and explain why navigation did not complete (Priority: P3)

作为部署操作员，我需要在导航失败时看到明确原因，并确保系统不会在站位不正确的情况下继续操作。

**Why this priority**: 可观测性和安全停止对于真实机器人部署是必要条件，否则失败会变成不可诊断的误操作风险。

**Independent Test**: 人为制造导航失败条件，例如规划器不可用、场景数据缺失或超过最大尝试次数，验证系统停止底盘、不给手臂放行，并输出终止原因。

**Acceptance Scenarios**:

1. **Given** required scene or position data is unavailable during navigation, **When** the system cannot obtain the information needed for the next decision, **Then** it stops the base, ends navigation as failed, and does not start manipulation.
2. **Given** the planner returns an invalid or unusable directive, **When** the system evaluates that response, **Then** it refuses the directive, keeps the base stopped, and reports the failure reason.
3. **Given** the configured navigation attempt limit is reached before arrival is confirmed, **When** the limit is hit, **Then** the system ends navigation with a failure outcome and blocks manipulation from starting.
4. **Given** a navigation session ends for any reason, **When** the operator inspects the session output, **Then** the operator can see the last planner decision, the executed movement, and the final termination reason.

### Edge Cases

- Forward scene data is stale or unavailable when a navigation-enabled task begins.
- The planner is temporarily unavailable or returns invalid responses for up to 3 consecutive cycles; the system stays stopped while retrying, those failures do not consume navigation steps, and navigation fails on the 4th consecutive failure.
- The operator does not explicitly confirm the navigation safety warning at task start.
- Robot position and heading data becomes unavailable between navigation cycles.
- The planner asks for movement outside configured speed limits; the system rejects that cycle, keeps the base stopped, records the rejection, and replans.
- The planner never confirms arrival even though navigation remains active.
- The operator disables navigation for a task prompt that still contains movement intent.
- Startup planner settings are incomplete or invalid for the current run.
- The robot starts close enough to the goal that no base movement is needed before handoff.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: When navigation is enabled and a task prompt is provided, the system MUST execute a navigation stage before starting the manipulation stage.
- **FR-001A**: When navigation is enabled, the system MUST present a safety warning and require explicit operator confirmation before any navigation movement is allowed to begin.
- **FR-002**: During navigation, the system MUST evaluate the task goal using the latest forward scene view and the robot's current position and heading.
- **FR-003**: The system MUST consult an external planner on each navigation cycle using the task prompt, current scene and position context, and recent navigation history, while keeping the base stopped whenever a planner decision is pending.
- **FR-004**: Planner decisions MUST support both incremental movement instructions and explicit arrival decisions.
- **FR-005**: The system MUST reject any planner decision whose requested movement exceeds the configured forward or turning speed limits, keep the base stopped for that cycle, record the rejection, and request a new planner decision on the next cycle.
- **FR-006**: The system MUST stop the base at the end of each navigation cycle before requesting another planner decision or handing off to manipulation.
- **FR-007**: The system MUST continue the navigation loop until either arrival is confirmed or the configured maximum number of usable planner decisions is reached.
- **FR-008**: The system MUST start manipulation only after navigation ends with an explicit arrival decision and the base has been commanded to remain stationary.
- **FR-009**: The system MUST prevent manipulation from starting if navigation ends because of planner unavailability, invalid planner output, missing required robot state, or reaching the maximum navigation attempts without an arrival decision.
- **FR-009A**: The system MUST retry planner queries up to 3 consecutive times when the planner is unavailable or returns an invalid response, keep the base stopped during those retries, and end navigation as failed on the 4th consecutive planner failure; those planner failures MUST NOT count toward the navigation step limit.
- **FR-010**: The system MUST allow the operator to disable navigation at task start and proceed directly to manipulation.
- **FR-011**: The system MUST allow operators to set planner access settings, navigation attempt limits, and movement limits when starting a job.
- **FR-012**: The system MUST record each navigation cycle's planner decision, executed movement, and final termination reason in a form operators can inspect during validation and troubleshooting.
- **FR-013**: The system MUST surface a clear status for navigation skipped, navigation succeeded, and navigation failed outcomes.
- **FR-014**: The system MUST preserve the original task objective so the downstream manipulation stage receives the same user intent after navigation completes.
- **FR-015**: Implementation and validation work for this feature MUST reference the locally stored TRACER 2.0 user manual so platform operating limits, interfaces, and safety constraints remain aligned with the documented robot behavior.

## Dependencies

- Local reference manual: `docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf`

### Key Entities *(include if feature involves data)*

- **Task Prompt**: The operator-provided description of the end-to-end job, including where the robot should go and what operation should happen after arrival.
- **Navigation Session**: A single pre-operation navigation attempt, including whether navigation is enabled, how many decision cycles have run, and whether the session ended in success, skip, or failure.
- **Scene and Position Snapshot**: The latest forward-facing scene view plus the robot's current position and heading used to decide the next navigation action.
- **Planner Decision**: A structured instruction from the external planner that either requests bounded robot movement or declares that the target operating position has been reached.
- **Handoff Outcome**: The final state produced by navigation that determines whether manipulation may start, is skipped, or is blocked.

## Assumptions

- Operators start the workflow in an environment where forward scene data and robot position data are already available to the system.
- The external planner is capable of determining whether the robot has reached a usable operating position from the provided task context and current scene.
- Manual teleoperation, obstacle avoidance policy changes, and manipulation policy changes are outside the scope of this feature.
- The manipulation stage keeps its current behavior except that it now waits for a navigation success or skip outcome before starting.
- Engineers implementing this feature are expected to consult the local TRACER 2.0 user manual for documented chassis behavior, operating limits, and safety notes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In at least 90% of trials where the goal area is reachable and the planner is available, the system reaches either a successful handoff or a clear failure outcome within the configured navigation attempt limit.
- **SC-002**: In 100% of successful navigation-assisted runs, manipulation begins only after the navigation stage has ended and the base has been commanded to a stationary state.
- **SC-003**: Operators can disable navigation and start manipulation-only execution within the same launch flow in 100% of bypass cases.
- **SC-004**: In 100% of failed navigation sessions, operators can identify why navigation stopped and what the last executed movement was from the system output.
- **SC-005**: During validation runs that require repositioning, operators observe chassis motion before manipulation begins in at least 95% of successful end-to-end executions.
