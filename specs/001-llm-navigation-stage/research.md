# Research: TRACER 2.0 Navigation-First Operation Flow

## Decision 1: Reuse the existing `RosOperator` sensing and base-control path

- **Decision**: The planner will read the latest front camera frame from `RosOperator.img_front_deque`, read odometry from `RosOperator.robot_base_deque`, and send base motion through `RosOperator.robot_base_publish()`.
- **Rationale**: `examples/piper_real/ros_oper.py` already subscribes to `/camera_f/color/image_raw` and `/odom_raw`, and already publishes `/cmd_vel`. Reusing those buffers keeps the feature aligned with the current deploy flow and avoids duplicate ROS subscriptions, extra synchronization logic, and another control process.
- **Alternatives considered**:
  - Add a separate ROS node or planner-specific subscribers: rejected because it duplicates transport paths that already exist in the deploy example.
  - Route chassis motion through `real_env.step()`: rejected because `real_env.py` keeps `use_robot_base=False`, and the feature must leave the manipulation action loop unchanged.

## Decision 2: Isolate navigation logic in a new planner module plus nested config

- **Decision**: Add `examples/piper_real/llm_planner.py` for the navigation loop and `examples/piper_real/planner_config.py` for planner settings, then instantiate the planner in `examples/piper_real/main.py` after environment creation and before `Runtime.run()`.
- **Rationale**: This preserves a clean separation between pre-manipulation navigation and OpenPI arm execution, keeps the new behavior local to the deploy example, and maps naturally onto tyro's nested dataclass CLI expansion.
- **Alternatives considered**:
  - Inline planner logic directly into `main.py`: rejected because it would mix orchestration, runtime wiring, LLM client logic, and ROS command validation in one file.
  - Put navigation logic inside `env.py`: rejected because the environment wrapper should stay focused on OpenPI observation/action semantics, not external planner orchestration.

## Decision 3: Use an OpenAI-compatible multimodal chat-completions contract with strict JSON validation

- **Decision**: Call the external planner through `openai.OpenAI(base_url=..., api_key=...).chat.completions.create(...)`, send the task prompt, odometry summary, recent navigation history, and front-camera image as a data URI, and require a JSON-only response with either a `move` or `stop` action.
- **Rationale**: The feature requirement explicitly calls for a local OpenAI-compatible API. Chat completions with multimodal content are the broadest-compatibility path across self-hosted vision-language servers. A strict JSON contract minimizes ambiguity in control decisions and keeps logging/debugging straightforward.
- **Alternatives considered**:
  - Use the OpenAI Responses API: rejected because OpenAI-compatible local servers most commonly emulate chat completions first.
  - Use a raw `requests` client: rejected because the `openai` SDK already handles the compatibility layer cleanly and reduces ad hoc protocol code.
  - Use text-only planner prompts: rejected because the feature depends on visual scene understanding.

## Decision 4: Treat safety as a runtime control gate, not a documentation-only warning

- **Decision**: When navigation is enabled, the runtime must display a safety warning and require explicit operator confirmation before any chassis motion begins. If confirmation is declined or absent, the navigation-enabled run fails before movement and does not hand off to manipulation.
- **Rationale**: The TRACER 2.0 manual states that the chassis does not provide autonomous anti-collision, anti-drop, or biological proximity warning functions and recommends operation in a relatively open area under operator awareness. Converting that into an explicit runtime gate makes the requirement testable and reduces deployment ambiguity.
- **Alternatives considered**:
  - Rely only on the manual and operator convention: rejected because it leaves a critical safety dependency outside the software workflow.
  - Show a warning without requiring confirmation: rejected because it does not create an enforceable stop point before movement.

## Decision 5: Use rejection-and-replan semantics for unsafe planner commands and bounded retries for planner outages

- **Decision**: Reject any planner command whose linear or angular velocity exceeds configured limits, keep the base stopped for that cycle, record the rejection, and request a new planner decision. Retry planner queries up to 3 consecutive times when responses are invalid or unavailable; fail navigation on the 4th consecutive failure. Those planner failures do not consume the navigation step budget, which only counts usable planner decisions.
- **Rationale**: This behavior matches the clarified spec and keeps control conservative. Rejecting overspeed commands is safer than silently clamping them, while a short retry window tolerates transient local model outages without creating an unbounded wait state.
- **Alternatives considered**:
  - Clamp overspeed commands and execute them anyway: rejected because it hides planner mistakes and changes command intent silently.
  - Fail immediately on the first invalid or unavailable planner response: rejected because local planner services can transiently fail and the feature already allows safe stop-and-retry behavior.
  - Count every planner query toward the navigation step budget: rejected because external service instability would prematurely exhaust navigation attempts even when no usable decision was received.

## Decision 6: Keep per-cycle movement short and observable

- **Decision**: Each usable `move` response carries `linear_x`, `angular_z`, and `duration`; if duration is absent, the planner adapter falls back to `PlannerConfig.default_duration`. After every move cycle, the runtime immediately publishes zero velocity before the next planner query or handoff.
- **Rationale**: Short bounded motion segments improve observability, align with the stop-between-cycles requirement, and give the planner fresh visual/odometry feedback before each new decision. A default duration preserves compatibility with slightly imperfect planner output while keeping local behavior predictable.
- **Alternatives considered**:
  - Allow continuous open-loop motion across planner cycles: rejected because it weakens the safety boundary and complicates validation.
  - Require every move response to include a valid duration with no fallback: rejected because it makes the integration less tolerant of otherwise-usable local planner responses.

## Decision 7: Compute yaw locally from odometry and log normalized planner outcomes each cycle

- **Decision**: Convert the latest `nav_msgs/Odometry` quaternion to `{x, y, yaw}` inside the planner module, track cycle history in memory, and log both the raw planner JSON and the normalized command or failure reason on every cycle.
- **Rationale**: The feature needs human-readable observability for validation (`/cmd_vel` plus planner logs), and yaw extraction can be implemented with standard quaternion math without adding a new ROS transform dependency.
- **Alternatives considered**:
  - Pass the raw quaternion directly to the planner: rejected because a normalized `{x, y, yaw}` summary is easier to reason about and easier to log.
  - Add a `tf`/`tf_transformations` dependency: rejected because the calculation is simple enough to keep local.

## Decision 8: Minimize dependency changes and keep verification deploy-focused

- **Decision**: Add `openai` to `examples/piper_real/requirements.in`, regenerate `examples/piper_real/requirements.txt`, and validate the feature primarily through deploy-mode ROS checks: planner logs, `/cmd_vel` output, navigation-before-manipulation behavior, and navigation bypass.
- **Rationale**: The current repository does not already contain a meaningful automated test harness for the `piper_real` ROS workflow, so the highest-value verification remains the real hardware/deploy path described in the feature spec. The dependency delta stays small and local to the deploy example.
- **Alternatives considered**:
  - Introduce a broader ROS simulation or end-to-end test harness in this feature: rejected because it is outside the requested scope and would add significant unrelated complexity.
  - Implement the planner client with custom HTTP code to avoid a dependency change: rejected because the `openai` SDK is the intended compatibility layer and is easier to maintain.
