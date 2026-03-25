# Replay Pure-Offline & Two-Layer LLM Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace replay mode with pure HDF5 data inspection (no inference server), and upgrade the LLM planner from single-navigation to a two-layer task decomposition + sequential execution architecture.

**Architecture:** New `TaskDecomposer` decomposes prompts into ordered navigate/manipulate subtasks via one-shot LLM call. Existing `LLMNavigationPlanner` executes individual navigate subtasks. `main.py` orchestrates the subtask loop. Shared JSON extraction utility avoids duplication.

**Tech Stack:** Python 3.10+, h5py, OpenAI client, tyro, ROS (rospy), openpi_client Runtime

**Spec:** `docs/superpowers/specs/2026-03-25-replay-and-two-layer-planner-design.md`

**Testing note:** This project has no automated test suite (testing is done via real robot deployment per CLAUDE.md). Verification steps use manual inspection and dry-run commands instead of pytest.

---

### Task 1: Extract shared JSON utility (`llm_utils.py`)

**Files:**
- Create: `examples/piper_real/llm_utils.py`
- Modify: `examples/piper_real/llm_planner.py:283-293`

- [ ] **Step 1: Create `llm_utils.py` with `extract_json_text`**

```python
"""Shared LLM response parsing utilities."""


def extract_json_text(raw_text: str) -> str:
    """Extract JSON object from an LLM response that may contain markdown fences."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return stripped[start : end + 1]
```

- [ ] **Step 2: Update `llm_planner.py` to import from shared utility**

In `examples/piper_real/llm_planner.py`, replace the `_extract_json_text` static method (lines 283-293) with an import:

```python
# At top of file, add:
from examples.piper_real.llm_utils import extract_json_text

# In query_llm method (line 100), change:
#   raw_json = self._extract_json_text(raw_text)
# to:
try:
    raw_json = extract_json_text(raw_text)
except ValueError as exc:
    raise PlannerResponseError(str(exc)) from exc

# Delete the _extract_json_text static method (lines 283-293)
```

- [ ] **Step 3: Verify no import errors**

Run: `python -c "from examples.piper_real import llm_planner; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add examples/piper_real/llm_utils.py examples/piper_real/llm_planner.py
git commit -m "refactor: extract shared extract_json_text into llm_utils.py"
```

---

### Task 2: Rewrite replay branch (pure offline HDF5 playback)

**Files:**
- Modify: `examples/piper_real/main.py:49-91`

- [ ] **Step 1: Replace the replay branch in `main.py`**

Replace lines 49-91 (the entire `if args.replay_dataset:` block) with:

```python
    # ── Replay mode: pure offline HDF5 data inspection ────────────────
    if args.replay_dataset:
        import h5py

        logging.info("Replay mode (offline): loading %s", args.replay_dataset)
        with h5py.File(args.replay_dataset, "r") as f:
            actions: np.ndarray = f["/action"][()]
            has_base_action = "base_action" in f
            base_actions: np.ndarray | None = (
                f["/base_action"][()] if has_base_action else None
            )

        num_steps = actions.shape[0]
        action_dim = actions.shape[1] if actions.ndim > 1 else 0

        for i in range(num_steps):
            arm = actions[i]
            arm_str = ", ".join(f"{v:.4f}" for v in arm[:14])
            if base_actions is not None and i < len(base_actions):
                base = base_actions[i]
                base_str = f"[{base[0]:.4f}, {base[1]:.4f}]"
            elif action_dim >= 16:
                base_str = f"[{arm[14]:.4f}, {arm[15]:.4f}]"
            else:
                base_str = "N/A"
            logging.info(
                "Replay step %d/%d -- arm: [%s] base: %s",
                i, num_steps, arm_str, base_str,
            )

        logging.info(
            "Replay complete: %d steps, action_dim=%d, has_base_action=%s",
            num_steps, action_dim, has_base_action,
        )
        return
```

- [ ] **Step 2: Remove unused replay imports**

The `replay_env` import inside the old replay branch is no longer needed. The file `replay_env.py` can remain in the repo but is now unused by `main.py`.

- [ ] **Step 3: Verify the replay branch parses correctly**

Run: `python -c "from examples.piper_real import main; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add examples/piper_real/main.py
git commit -m "feat(replay): replace inference-based replay with pure offline HDF5 playback"
```

---

### Task 3: Remove policy-driven base velocity from env/real_env

**Files:**
- Modify: `examples/piper_real/env.py:18-27,85-95`
- Modify: `examples/piper_real/real_env.py:83-104,155-174,222-257,270-286`

- [ ] **Step 1: Update `env.py` — remove base params, add `set_prompt`, truncate action**

In `examples/piper_real/env.py`:

a) Remove `use_robot_base`, `max_base_linear_vel`, `max_base_angular_vel` from `__init__` params and from the `make_real_env` call:

```python
class PiperRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        prompt: str = "",
    ) -> None:
        self._env = _real_env.make_real_env(
            init_node=True,
            reset_position=reset_position,
        )
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None
        self.save_obs = True
        self.frame_cnt = 0
        if self.save_obs:
            self.saver = _obs_saver()
```

b) Remove the unused imports at the top of env.py:

```python
# Remove these two lines:
# from examples.piper_real.base_safety import TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S
# from examples.piper_real.base_safety import TRACER_MANUAL_MAX_LINEAR_VEL_MPS
```

c) Add `set_prompt` method after the `ros_operator` property:

```python
    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
```

d) In `apply_action`, truncate action to 14 dims before passing to `step()`:

```python
    @override
    def apply_action(self, action: dict) -> None:
        if self.save_obs and "actions" in action:
            self.saver.save_output_action_to_csv(action["actions"])
            print(f"action: {action['actions']}")
        print("main action apply")
        stop_flag = action.get("STOP", False)
        print(f"STOP_SIGNAL: {stop_flag}")
        if "actions" in action:
            raw = action["actions"]
            truncated = raw[:14] if len(raw) > 14 else raw
            self._ts = self._env.step(truncated, STOP=stop_flag)
        else:
            self._ts = self._env.step(None, STOP=stop_flag)
```

- [ ] **Step 2: Update `real_env.py` — remove base velocity handling**

In `examples/piper_real/real_env.py`:

a) Remove from `PiperRealEnv.__init__` params: `use_robot_base`, `max_base_linear_vel`, `max_base_angular_vel`. Remove `ros_config["use_robot_base"] = use_robot_base` and `self._max_base_linear_vel` / `self._max_base_angular_vel` assignments:

```python
    def __init__(
        self,
        init_node,
        *,
        reset_pos: Optional[List[float]] = None,
        setup_robots: bool = False,
    ):
        if init_node:
            rospy.init_node('joint_state_publisher_pi0_debug', anonymous=True)
            self.spin_thread = Thread(target=self.spin)
            self.spin_thread.start()
        self._reset_pos = reset_pos
        self.ros_operator = _ros_oper.RosOperator(ros_config)
        self.rate = rospy.Rate(ros_config["publish_rate"])
        self.pre_action = np.zeros(ros_config['state_dim'])
```

b) Delete the entire `_extract_base_velocity` method (lines 155-174).

c) In `step()`, remove the `use_robot_base` block (lines 252-254):

```python
    def step(self, action, STOP=False):
        interp_actions = None

        if STOP:
            base_safety.stop_base(self.ros_operator)
            print("[STOP] skipping action publish.")
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=self.get_reward(),
                discount=None,
                observation=self.get_observation()
            )

        if ros_config["use_actions_interpolation"]:
            print(f"use_actions_interpolation")
            interp_actions = interpolate_action(ros_config, self.pre_action, action)
        else:
            interp_actions = action[np.newaxis, :]

        for act in interp_actions:
            state_len = int(len(act) / 2)
            left_action = act[:state_len]
            right_action = act[state_len:]

            if not ros_config["disable_puppet_arm"]:
                self.ros_operator.puppet_arm_publish(left_action, right_action)

            self.rate.sleep()

        self.pre_action = action.copy()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )
```

d) Update `make_real_env` to remove base params:

```python
def make_real_env(
    init_node,
    *,
    reset_position: Optional[List[float]] = None,
    setup_robots: bool = True,
) -> PiperRealEnv:
    return PiperRealEnv(
        init_node,
        reset_pos=reset_position,
        setup_robots=setup_robots,
    )
```

e) Keep the `base_safety` import — it is still used by the STOP handler in `step()`. Remove `import logging` only if it was solely used by `_extract_base_velocity` (check first).

- [ ] **Step 3: Verify imports still work**

Run: `python -c "from examples.piper_real import env; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add examples/piper_real/env.py examples/piper_real/real_env.py
git commit -m "refactor: remove policy-driven base velocity from env and real_env

Manipulate phase now always outputs 14-dim arm-only actions.
Base movement is exclusively handled by LLMNavigationPlanner."
```

---

### Task 4: Add new validation rules to `main.py`

**Files:**
- Modify: `examples/piper_real/main.py:39-47`

- [ ] **Step 1: Replace the validation block at the top of `main()`**

Replace lines 42-47 (the `navigation_only` coercion and mutual-exclusion checks) with explicit validation:

```python
    if args.navigation_only and not args.use_llm_planner:
        logging.error("--navigation-only requires --use-llm-planner.")
        return

    if args.use_robot_base and not args.use_llm_planner:
        logging.error("--use-robot-base requires --use-llm-planner.")
        return

    if args.navigation_only and args.replay_dataset:
        logging.error("--navigation-only and --replay-dataset are mutually exclusive.")
        return

    if args.use_llm_planner and args.replay_dataset:
        logging.error("--use-llm-planner and --replay-dataset are mutually exclusive.")
        return
```

- [ ] **Step 2: Verify import parses**

Run: `python -c "from examples.piper_real import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/piper_real/main.py
git commit -m "feat: add explicit validation rules for flag combinations

--use-robot-base and --navigation-only now require --use-llm-planner.
--use-llm-planner and --replay-dataset are mutually exclusive."
```

---

### Task 5: Create `TaskDecomposer`

**Files:**
- Create: `examples/piper_real/task_decomposer.py`

- [ ] **Step 1: Create the file**

```python
"""Task decomposition: one-shot LLM call to split a prompt into subtasks."""

import dataclasses
import json
import logging
from typing import Any

from openai import OpenAI

from examples.piper_real.llm_utils import extract_json_text
from examples.piper_real.planner_config import PlannerConfig

_MAX_SUBTASKS = 10
_MAX_ATTEMPTS = 4

_VALID_TYPES = {"navigate", "manipulate"}

_SYSTEM_PROMPT = (
    "You are a task planner for a mobile manipulation robot. "
    "Given a task description, decompose it into an ordered list of subtasks. "
    "Each subtask is either 'navigate' (move the robot base to a location) "
    "or 'manipulate' (use the robot arms to interact with an object). "
    "Return JSON only with no markdown. "
    "Format: {\"subtasks\": [{\"type\": \"navigate\"|\"manipulate\", \"prompt\": \"...\"}]}. "
    "Rules: "
    "- If the task requires moving to a location first, start with a navigate subtask. "
    "- If the robot is already at the right location, use only manipulate subtasks. "
    "- A single manipulate subtask with no navigation is valid. "
    "- Keep the list concise; do not exceed 10 subtasks. "
    "- Each prompt should be a clear, self-contained instruction."
)


class DecompositionError(RuntimeError):
    """Raised when task decomposition fails after all retries."""


@dataclasses.dataclass
class Subtask:
    type: str   # "navigate" | "manipulate"
    prompt: str


class TaskDecomposer:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def decompose(self, task_prompt: str) -> list[Subtask]:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._attempt_decompose(task_prompt)
            except (ValueError, KeyError, TypeError) as exc:
                last_error = exc
                logging.warning(
                    "Decomposition attempt %d/%d failed: %s",
                    attempt + 1, _MAX_ATTEMPTS, exc,
                )
        raise DecompositionError(
            f"Task decomposition failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _attempt_decompose(self, task_prompt: str) -> list[Subtask]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ],
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            raw_text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        else:
            raw_text = str(content)

        raw_json = extract_json_text(raw_text.strip())
        payload = json.loads(raw_json)

        subtasks_raw = payload["subtasks"]
        if not isinstance(subtasks_raw, list) or len(subtasks_raw) == 0:
            raise ValueError("subtasks must be a non-empty list")
        if len(subtasks_raw) > _MAX_SUBTASKS:
            raise ValueError(
                f"subtask count {len(subtasks_raw)} exceeds maximum {_MAX_SUBTASKS}"
            )

        result: list[Subtask] = []
        for i, entry in enumerate(subtasks_raw):
            st_type = entry.get("type", "")
            st_prompt = entry.get("prompt", "")
            if st_type not in _VALID_TYPES:
                raise ValueError(
                    f"subtask[{i}].type must be one of {_VALID_TYPES}, got {st_type!r}"
                )
            if not st_prompt.strip():
                raise ValueError(f"subtask[{i}].prompt must be non-empty")
            result.append(Subtask(type=st_type, prompt=st_prompt.strip()))

        logging.info(
            "Task decomposition: %s",
            json.dumps(
                {"subtasks": [{"type": s.type, "prompt": s.prompt} for s in result]},
                ensure_ascii=False,
            ),
        )
        return result
```

- [ ] **Step 2: Verify import**

Run: `python -c "from examples.piper_real.task_decomposer import TaskDecomposer, Subtask; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/piper_real/task_decomposer.py
git commit -m "feat: add TaskDecomposer for two-layer LLM planner architecture"
```

---

### Task 6: Rewrite `main.py` subtask execution loop

**Files:**
- Modify: `examples/piper_real/main.py` (lines 93-171, the entire post-validation flow)

This is the largest change. The entire flow after replay and validation is replaced with the two-layer subtask loop.

- [ ] **Step 1: Add new imports at top of `main.py`**

```python
from examples.piper_real import task_decomposer as _task_decomposer
```

- [ ] **Step 2: Rewrite the main flow after validation/replay**

Replace everything from line 93 (after the replay `return`) through end of `main()` with:

```python
    # ── Stationary manipulation (no LLM planner) ─────────────────────
    if not args.use_llm_planner:
        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

        if args.save_log:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=args.prompt,
        )

        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=50,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
        return

    # ── Two-layer LLM planner ────────────────────────────────────────
    args.planner.validate_service_config()

    if not prompt:
        logging.info("LLM planner enabled but prompt is empty; running stationary manipulation.")
        # Fall through — create server connection and run a single manipulation episode
        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

        if args.save_log:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=args.prompt,
        )
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=50,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
        return

    # Step 0: Validate motion limits early to avoid wasting an LLM call
    if args.use_robot_base:
        args.planner.validate_motion_limits()

    # Step 1: Decompose task
    decomposer = _task_decomposer.TaskDecomposer(args.planner)
    try:
        subtask_list = decomposer.decompose(prompt)
    except _task_decomposer.DecompositionError as exc:
        logging.error("Task decomposition failed: %s", exc)
        return

    has_navigate = any(s.type == "navigate" for s in subtask_list)
    has_manipulate = any(s.type == "manipulate" for s in subtask_list)
    needs_server = has_manipulate and not args.navigation_only

    # Step 2: Safety confirmation (once, if base motion requested)
    if args.use_robot_base and has_navigate:
        if not _base_safety.confirm_base_motion_safety(
            prompt,
            use_llm_planner=True,
            use_robot_base=False,  # pass False to suppress misleading "policy-driven base control" label
        ):
            logging.error("Base motion aborted before execution.")
            return

    # Step 4: Create inference server connection if needed
    ws_client_policy = None
    metadata = {}
    if needs_server:
        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        metadata = ws_client_policy.get_server_metadata()
        logging.info("Server metadata: %s", metadata)

    # Step 5: Create shared environment if needed
    environment = None
    if needs_server:
        if args.save_log:
            _logger.InputJointStateLogger()
            _logger.OutputJointStateLogger()

        environment = _env.PiperRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=prompt,
        )

    # Step 6: Create navigation planner if needed
    planner = None
    if args.use_robot_base and has_navigate:
        if environment is not None:
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)
        else:
            # Navigation-only mode: need ROS for base movement but no manipulation env
            # Create a minimal environment just for ros_operator access
            environment = _env.PiperRealEnvironment(
                reset_position=None,
                prompt=prompt,
            )
            planner = _llm_planner.LLMNavigationPlanner(environment.ros_operator, args.planner)

    # Step 7: Execute subtask loop
    try:
        for idx, subtask in enumerate(subtask_list):
            logging.info(
                "Executing subtask %d/%d [%s]: %s",
                idx + 1, len(subtask_list), subtask.type, subtask.prompt,
            )

            if subtask.type == "navigate":
                if args.use_robot_base:
                    if not planner.run(task_prompt=subtask.prompt):
                        _base_safety.stop_base(environment.ros_operator)
                        logging.error(
                            "Navigation failed at subtask %d/%d; aborting.",
                            idx + 1, len(subtask_list),
                        )
                        return
                    logging.info("Navigate subtask %d/%d succeeded.", idx + 1, len(subtask_list))
                else:
                    logging.info("Navigate (dry-run): %s", subtask.prompt)

            elif subtask.type == "manipulate":
                if args.navigation_only:
                    logging.info("Manipulate (skipped): %s", subtask.prompt)
                    continue

                assert ws_client_policy is not None, "manipulate subtask requires server connection"
                environment.set_prompt(subtask.prompt)
                runtime = _runtime.Runtime(
                    environment=environment,
                    agent=_policy_agent.PolicyAgent(
                        policy=action_chunk_broker.ActionChunkBroker(
                            policy=ws_client_policy,
                            action_horizon=args.action_horizon,
                        )
                    ),
                    subscribers=[],
                    max_hz=50,
                    num_episodes=args.num_episodes,
                    max_episode_steps=args.max_episode_steps,
                )
                runtime.run()
                logging.info("Manipulate subtask %d/%d completed.", idx + 1, len(subtask_list))

        logging.info("All subtasks completed successfully.")
    finally:
        if args.use_robot_base and environment is not None:
            _base_safety.stop_base(environment.ros_operator)
```

- [ ] **Step 3: Clean up unused imports**

Remove any imports that are no longer used after the rewrite. The `base_motion_requested` / `navigation_requested` variables and the old single-navigation flow are all gone.

- [ ] **Step 4: Verify the full module parses**

Run: `python -c "from examples.piper_real import main; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add examples/piper_real/main.py
git commit -m "feat: implement two-layer subtask execution loop in main.py

TaskDecomposer splits the prompt into navigate/manipulate subtasks.
Each navigate subtask is executed by LLMNavigationPlanner (or dry-run).
Each manipulate subtask gets its own Runtime run with updated prompt.
WebsocketClientPolicy only created when manipulation subtasks exist."
```

---

### Task 7: Update `docs/deploy.md`

**Files:**
- Modify: `docs/deploy.md`

- [ ] **Step 1: Update Section 0 (Safety prerequisites)**

Update the base-motion trigger conditions (lines 17-22) from:

> - `--use-llm-planner` 且 `--prompt` 非空
> - `--use-robot-base`

To:

> - `--use-llm-planner` 且 `--use-robot-base` 且 `--prompt` 非空

This reflects the new architecture where `--use-llm-planner` alone does NOT trigger base motion — only `--use-robot-base` does.

- [ ] **Step 2: Update Section 3 (CLI switches)**

Replace the current flag table and examples with the new semantics:

```markdown
## 3. CLI 开关

`examples/piper_real/main.py` 提供以下顶层开关：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use-llm-planner` | `False` | 启用两层架构：先拆解任务再按 subtask 执行 |
| `--use-robot-base` | `False` | navigate subtask 是否实际移动底盘（需要 `--use-llm-planner`） |
| `--navigation-only` | `False` | 只执行 navigate subtask，跳过 manipulate（需要 `--use-llm-planner`） |

Flag 组合规则：

- `--use-robot-base` 需要 `--use-llm-planner`
- `--navigation-only` 需要 `--use-llm-planner`
- `--replay-dataset` 与 `--use-llm-planner` 互斥
- `--replay-dataset` 与 `--navigation-only` 互斥
```

Update the CLI examples:

```markdown
```bash
# 仅手臂操作（默认行为，不使用 LLM planner）
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --prompt "turn on the water tap."

# LLM 拆解 + 导航移动 + 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --use-robot-base \
  --prompt "移动到桌子旁边拿起红色杯子"

# LLM 拆解 + 导航仅打印（dry-run）+ 手臂操作
python -m examples.piper_real.main \
  --host 192.168.3.101 --port 8000 \
  --use-llm-planner \
  --prompt "移动到桌子旁边拿起红色杯子"

# 仅导航（实际移动底盘，跳过操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --use-robot-base \
  --navigation-only \
  --prompt "依次移动到厨房和客厅"

# 仅导航 dry-run（仅打印计划，不移动不操作）
python -m examples.piper_real.main \
  --use-llm-planner \
  --navigation-only \
  --prompt "依次移动到厨房和客厅"
```
```

- [ ] **Step 3: Update Section 5 (LLM planner)**

Replace the single-navigation description with the two-layer architecture:

```markdown
## 5. LLM 两层任务架构（`--use-llm-planner`）

启用后，系统先调用 LLM 将完整 prompt 拆解为有序的 subtask 列表（navigate + manipulate），然后按序执行。

### 两层架构

1. **TaskDecomposer**: 一次性 LLM 调用，将 prompt 拆解为 subtask 列表。
2. **LLMNavigationPlanner**: 执行单个 navigate subtask 的多步导航循环。

### 执行流程

1. LLM 拆解 prompt 为 `[{type: "navigate"|"manipulate", prompt: "..."}]`。
2. 如果 `--use-robot-base` 且有 navigate subtask，要求操作员输入 `yes` 确认。
3. 按序执行每个 subtask：
   - **navigate**: `--use-robot-base` 时实际移动底盘；否则仅打印。
   - **manipulate**: `--navigation-only` 时跳过；否则启动策略推理。
4. navigate 失败时终止整个任务，不执行后续 subtask。
5. 每个 manipulate subtask 独立运行一次策略推理。
```

- [ ] **Step 4: Update Section 6 (policy base control)**

Replace with a note that policy-driven base control during manipulation has been removed:

```markdown
## 6. 操作阶段底盘行为

操作（manipulate）阶段策略输出固定为 14 维（仅手臂关节），不包含底盘控制。底盘移动仅在 navigate subtask 中由 LLMNavigationPlanner 执行。

如果策略模型输出超过 14 维，多余维度会被截断。
```

- [ ] **Step 5: Update Section 8 (replay)**

Update to reflect the new pure-offline behavior:

```markdown
## 8. 离线回放调试（`--replay-dataset`）

使用已有 HDF5 数据集，逐步打印 ground-truth action，不需要推理服务器、实机或 ROS。

```bash
python -m examples.piper_real.main \
  --replay-dataset /home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5
```

回放模式下：

- 不连接推理服务器，不需要 `--host` 和 `--port`。
- 从 HDF5 文件逐帧打印 action（arm 14 维 + base 2 维，如有）。
- 回放结束后输出汇总（总步数、action 维度、是否包含 base_action）。
- `--use-llm-planner` 和 `--replay-dataset` 互斥。
```

- [ ] **Step 6: Commit**

```bash
git add docs/deploy.md
git commit -m "docs: update deploy guide for two-layer planner and pure offline replay"
```

---

### Task 8: End-to-end verification

- [ ] **Step 1: Verify all imports work**

```bash
python -c "
from examples.piper_real import main
from examples.piper_real import llm_planner
from examples.piper_real import task_decomposer
from examples.piper_real import env
from examples.piper_real import llm_utils
print('All imports OK')
"
```

- [ ] **Step 2: Verify replay mode with a real HDF5 file (if available)**

```bash
python -m examples.piper_real.main \
  --replay-dataset /home/agilex/rhos_cobot/ocl_data/turn_on_off_tap/episode_0.hdf5
```

Expected: prints each step's arm/base action values, then summary. No server connection attempted.

- [ ] **Step 3: Verify flag validation errors**

```bash
# Should error: --use-robot-base without --use-llm-planner
python -c "
from examples.piper_real.main import Args, main
args = Args(use_robot_base=True)
main(args)
" 2>&1 | grep "requires --use-llm-planner"

# Should error: --navigation-only without --use-llm-planner
python -c "
from examples.piper_real.main import Args, main
args = Args(navigation_only=True)
main(args)
" 2>&1 | grep "requires --use-llm-planner"
```

- [ ] **Step 4: Commit verification notes (optional)**

No code changes; this step confirms everything works together.
