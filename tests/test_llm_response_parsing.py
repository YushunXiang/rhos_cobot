from pathlib import Path
from types import SimpleNamespace


def test_extract_message_json_text_reads_reasoning_when_content_is_none():
    from examples.piper_real.llm_utils import extract_message_json_text

    message = {
        "content": None,
        "reasoning": '{"subtasks":[{"type":"navigate","prompt":"Move to the table"}]}',
    }

    raw_text, raw_json = extract_message_json_text(message)

    assert raw_text == message["reasoning"]
    assert raw_json == message["reasoning"]


def test_extract_message_json_text_skips_invalid_json_examples_before_valid_payload():
    from examples.piper_real.llm_utils import extract_message_json_text

    message = {
        "content": (
            "Thinking Process:\n"
            "Format: {\"subtasks\": [{\"type\": \"navigate\"|\"manipulate\", \"prompt\": \"...\"}]}\n"
            "{\"subtasks\":[{\"type\":\"navigate\",\"prompt\":\"Move to the sink\"}]}"
        ),
    }

    raw_text, raw_json = extract_message_json_text(message)

    assert raw_text == message["content"]
    assert raw_json == '{"subtasks":[{"type":"navigate","prompt":"Move to the sink"}]}'


def test_task_decomposer_accepts_reasoning_only_responses(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer(PlannerConfig(base_url="http://unused", model="test"))
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning=(
                        '{"subtasks":[{"type":"navigate","prompt":"Move to the table"},'
                        '{"type":"manipulate","prompt":"Pick up the red cup"}]}'
                    ),
                )
            )
        ]
    )
    monkeypatch.setattr(decomposer.client.chat.completions, "create", lambda **_kwargs: response)

    subtasks = decomposer.decompose("Move to the table and pick up the red cup.")

    assert [(subtask.type, subtask.prompt) for subtask in subtasks] == [
        ("navigate", "Move to the table"),
        ("manipulate", "Pick up the red cup"),
    ]


def test_task_decomposer_includes_ordered_task_context(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer(PlannerConfig(base_url="http://unused", model="test"))
    recorded: dict[str, object] = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"subtasks":['
                        '{"type":"manipulate","prompt":"Pick up the center plate from the sink."},'
                        '{"type":"manipulate","prompt":"Wash the plate under the running water."}'
                        "]}"
                    ),
                    reasoning=None,
                )
            )
        ]
    )

    def _fake_create(**kwargs):
        recorded.update(kwargs)
        return response

    monkeypatch.setattr(decomposer.client.chat.completions, "create", _fake_create)

    subtasks = decomposer.decompose(
        "Clean the plate and start the sandwich.",
        ordered_task_spec_text=(
            "1. Pick up the center plate\n"
            "2. Turn on the faucet\n"
            "3. Wash the plate"
        ),
        working_memory_text="Task progress: 0/2 (0%)",
        stage_estimate_text='{"current_subtask":"Pick up the center plate"}',
    )

    assert [(subtask.type, subtask.prompt) for subtask in subtasks] == [
        ("manipulate", "Pick up the center plate"),
        ("manipulate", "Wash the plate"),
    ]
    assert "Ordered task context" in recorded["messages"][1]["content"]
    assert "Working memory" in recorded["messages"][1]["content"]
    assert "Current stage estimate" in recorded["messages"][1]["content"]


def test_task_decomposer_disables_thinking_and_limits_tokens(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.task_decomposer import TaskDecomposer

    recorded: dict[str, object] = {}
    decomposer = TaskDecomposer(
        PlannerConfig(
            base_url="http://unused",
            model="test",
            task_decomposer_enable_thinking=False,
            task_decomposer_max_tokens=64,
        )
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"subtasks":[{"type":"navigate","prompt":"Move to the table"}]}',
                    reasoning=None,
                )
            )
        ]
    )

    def _fake_create(**kwargs):
        recorded.update(kwargs)
        return response

    monkeypatch.setattr(decomposer.client.chat.completions, "create", _fake_create)

    subtasks = decomposer.decompose("Move to the table.")

    assert [(subtask.type, subtask.prompt) for subtask in subtasks] == [
        ("navigate", "Move to the table")
    ]
    assert recorded["max_tokens"] == 64
    assert recorded["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_replay_navigation_only_falls_back_to_single_navigate_subtask(monkeypatch):
    from examples.piper_real.main import Args
    from examples.piper_real.main import _run_replay_planner
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real import replay_env as replay_env_mod
    from examples.piper_real import replay_planner as replay_planner_mod
    from examples.piper_real import task_decomposer as task_decomposer_mod

    recorded: dict[str, object] = {}

    class FakeReplayEnvironment:
        def __init__(self, dataset_path: str, prompt: str, max_steps: int | None = None) -> None:
            recorded["dataset_path"] = dataset_path
            recorded["prompt"] = prompt
            recorded["max_steps"] = max_steps
            recorded["closed"] = False
            self.num_steps = 10
            self.front_camera_name = "cam_high"
            self.camera_names = ("cam_high",)
            self.fps = 25.0

        def close(self) -> None:
            recorded["closed"] = True

        def get_cursor(self) -> int:
            return 0

    class FakeTaskDecomposer:
        def __init__(self, _config) -> None:
            pass

        def decompose(self, _prompt: str):
            raise task_decomposer_mod.DecompositionError("empty subtasks")

    class FakeOfflineReplayNavigationPlanner:
        def __init__(self, _environment, _config, on_step_callback=None) -> None:
            self.current_step = 3
            self.on_step_callback = on_step_callback

        def run(self, task_prompt: str) -> bool:
            recorded.setdefault("planner_prompts", []).append(task_prompt)
            return True

    monkeypatch.setattr(replay_env_mod, "ReplayEnvironment", FakeReplayEnvironment)
    monkeypatch.setattr(task_decomposer_mod, "TaskDecomposer", FakeTaskDecomposer)
    monkeypatch.setattr(
        replay_planner_mod,
        "OfflineReplayNavigationPlanner",
        FakeOfflineReplayNavigationPlanner,
    )

    args = Args(
        replay_dataset="/tmp/episode_4.hdf5",
        use_llm_planner=True,
        navigation_only=True,
        skip_server_checks=True,
        planner=PlannerConfig(base_url="http://unused", model="test"),
    )
    prompt = "long-horizon replay mock validation"

    _run_replay_planner(args, prompt)

    assert recorded["planner_prompts"] == [prompt]
    assert recorded["closed"] is True


def test_replay_manipulation_planner_accepts_reasoning_only_responses(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner

    class FakeReplayEnvironment:
        num_steps = 2
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

    replanner = ReplayManipulationPromptPlanner(
        FakeReplayEnvironment(),
        PlannerConfig(base_url="http://unused", model="test"),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning='{"action":"continue","prompt":"grasp the plate rim","reason":"object aligned"}',
                )
            )
        ]
    )
    monkeypatch.setattr(replanner.client.chat.completions, "create", lambda **_kwargs: response)

    decision = replanner.plan(
        task_prompt="pick up the plate",
        current_policy_prompt="pick up the plate",
        executed_policy_steps=0,
        prompt_history=[],
    )

    assert decision.action == "continue"
    assert decision.prompt == "grasp the plate rim"


def test_replay_manipulation_planner_disables_thinking_and_limits_tokens(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner

    class FakeReplayEnvironment:
        num_steps = 2
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

    recorded: dict[str, object] = {}
    replanner = ReplayManipulationPromptPlanner(
        FakeReplayEnvironment(),
        PlannerConfig(
            base_url="http://unused",
            model="test",
            manipulation_replanner_enable_thinking=False,
            manipulation_replanner_max_tokens=64,
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning='{"action":"complete","reason":"done"}',
                )
            )
        ]
    )

    def _fake_create(**kwargs):
        recorded.update(kwargs)
        return response

    monkeypatch.setattr(replanner.client.chat.completions, "create", _fake_create)

    decision = replanner.plan(
        task_prompt="pick up the plate",
        current_policy_prompt="pick up the plate",
        executed_policy_steps=0,
        prompt_history=[],
    )

    assert decision.action == "complete"
    assert recorded["max_tokens"] == 64
    assert recorded["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_replay_manipulation_planner_includes_ordered_task_memory_context(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner

    class FakeReplayEnvironment:
        num_steps = 2
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeTaskMemoryRuntime:
        def build_context(self) -> dict[str, str]:
            return {
                "ordered_task_spec_text": "Ordered subtask list:\n1. Pick up plate",
                "working_memory_text": "Task progress: 0/1 (0%)",
                "stage_estimate_text": '{"current_subtask":"Pick up plate"}',
            }

    recorded: dict[str, object] = {}
    replanner = ReplayManipulationPromptPlanner(
        FakeReplayEnvironment(),
        PlannerConfig(base_url="http://unused", model="test"),
        task_memory_runtime=FakeTaskMemoryRuntime(),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"action":"complete","reason":"done"}',
                    reasoning=None,
                )
            )
        ]
    )

    def _fake_create(**kwargs):
        recorded.update(kwargs)
        return response

    monkeypatch.setattr(replanner.client.chat.completions, "create", _fake_create)

    decision = replanner.plan(
        task_prompt="pick up the plate",
        current_policy_prompt="pick up the plate",
        executed_policy_steps=0,
        prompt_history=[],
    )

    assert decision.action == "complete"
    user_content = recorded["messages"][1]["content"]
    assert any(
        item["type"] == "text" and "Ordered task context" in item["text"]
        for item in user_content
    )


def test_ordered_task_spec_matches_decomposer_prompts_without_terminal_punctuation():
    from examples.piper_real.replay_task_memory import OrderedTaskSpec

    task_spec = OrderedTaskSpec(
        name="tap",
        total_task="tap",
        subtasks=[
            "Turn on the water tap.",
            "Turn off the water tap.",
        ],
    )

    assert task_spec.subtask_index("turn on the water tap") == 0
    assert task_spec.subtask_index("turn off the water tap") == 1


def test_replay_manipulation_planner_rejects_prompt_for_wrong_ordered_subtask(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner
    from examples.piper_real.replay_task_memory import OrderedTaskSpec
    from examples.piper_real.replay_task_memory import TaskStageDecision

    class FakeReplayEnvironment:
        num_steps = 2
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeTaskMemoryRuntime:
        def __init__(self) -> None:
            self.task_spec = OrderedTaskSpec(
                name="tap",
                total_task="tap",
                subtasks=[
                    "Turn on the water tap.",
                    "Turn off the water tap.",
                ],
            )
            self._last_decision = TaskStageDecision(
                current_subtask="Turn on the water tap.",
                current_subtask_index=0,
                completed_subtasks=[],
                next_subtask="Turn off the water tap.",
                confidence=0.95,
                evidence="tap is off",
                memory_update="tap is off",
                state_summary="tap is off",
            )

        def build_context(self) -> dict[str, str]:
            return {
                "ordered_task_spec_text": self.task_spec.as_prompt_text(),
                "working_memory_text": "Task progress: 0/2 (0%)",
                "stage_estimate_text": '{"current_subtask":"Turn on the water tap."}',
            }

    replanner = ReplayManipulationPromptPlanner(
        FakeReplayEnvironment(),
        PlannerConfig(base_url="http://unused", model="test"),
        task_memory_runtime=FakeTaskMemoryRuntime(),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"action":"continue","prompt":"turn on the water tap","reason":"regressed"}',
                    reasoning=None,
                )
            )
        ]
    )
    monkeypatch.setattr(replanner.client.chat.completions, "create", lambda **_kwargs: response)

    decision = replanner.plan(
        task_prompt="turn off the water tap",
        current_policy_prompt="turn off the water tap",
        executed_policy_steps=0,
        prompt_history=[],
    )

    assert decision.action == "continue"
    assert decision.prompt == "turn off the water tap"
    assert "different ordered subtask" in decision.reason


def test_replay_manipulation_planner_rejects_complete_before_stage_confirms_target(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_manipulation_planner import ReplayManipulationPromptPlanner
    from examples.piper_real.replay_task_memory import OrderedTaskSpec
    from examples.piper_real.replay_task_memory import TaskStageDecision

    class FakeReplayEnvironment:
        num_steps = 2
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeTaskMemoryRuntime:
        def __init__(self) -> None:
            self.task_spec = OrderedTaskSpec(
                name="tap",
                total_task="tap",
                subtasks=[
                    "Turn on the water tap.",
                    "Turn off the water tap.",
                ],
            )
            self._last_decision = TaskStageDecision(
                current_subtask="Turn on the water tap.",
                current_subtask_index=0,
                completed_subtasks=[],
                next_subtask="Turn off the water tap.",
                confidence=0.95,
                evidence="tap is off",
                memory_update="tap is off",
                state_summary="tap is off",
            )

        def build_context(self) -> dict[str, str]:
            return {
                "ordered_task_spec_text": self.task_spec.as_prompt_text(),
                "working_memory_text": "Task progress: 0/2 (0%)",
                "stage_estimate_text": '{"current_subtask":"Turn on the water tap."}',
            }

    replanner = ReplayManipulationPromptPlanner(
        FakeReplayEnvironment(),
        PlannerConfig(base_url="http://unused", model="test"),
        task_memory_runtime=FakeTaskMemoryRuntime(),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"action":"complete","reason":"turn on is done"}',
                    reasoning=None,
                )
            )
        ]
    )
    monkeypatch.setattr(replanner.client.chat.completions, "create", lambda **_kwargs: response)

    decision = replanner.plan(
        task_prompt="turn off the water tap",
        current_policy_prompt="turn off the water tap",
        executed_policy_steps=16,
        prompt_history=[],
    )

    assert decision.action == "continue"
    assert decision.prompt == "turn off the water tap"
    assert "ordered stage did not confirm" in decision.reason


def test_replay_ordered_task_memory_runtime_caches_current_step(monkeypatch):
    from examples.piper_real.planner_config import PlannerConfig
    from examples.piper_real.replay_task_memory import ReplayOrderedTaskMemoryRuntime

    class FakeReplayEnvironment:
        num_steps = 4
        camera_names = ("cam_high",)

        def get_cursor(self) -> int:
            return 0

        def get_image(self, _cam_name: str, _idx: int):
            import numpy as np

            return np.zeros((8, 8, 3), dtype=np.uint8)

        def get_state(self, _idx: int):
            import numpy as np

            return np.zeros(14, dtype=np.float32)

        def get_ground_truth_action(self, _idx: int):
            import numpy as np

            return np.zeros(14, dtype=np.float32)

    task_spec_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "episode4_plate_wash_sandwich.task_spec.json"
    )
    runtime = ReplayOrderedTaskMemoryRuntime(
        FakeReplayEnvironment(),
        PlannerConfig(
            base_url="http://unused",
            model="test",
            task_spec_path=str(task_spec_path),
            task_memory_max_entries=4,
        ),
    )
    recorded = {"calls": 0}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning=(
                        '{"current_subtask":"Pick up the center plate",'
                        '"current_subtask_index":0,'
                        '"completed_subtasks":[],'
                        '"next_subtask":"Turn on the faucet",'
                        '"confidence":0.9,'
                        '"evidence":"The gripper is moving toward the center plate.",'
                        '"memory_update":"The plate is still on the counter.",'
                        '"state_summary":"The center plate is on the counter near the sink."}'
                    ),
                )
            )
        ]
    )

    def _fake_create(**kwargs):
        recorded["calls"] += 1
        recorded["kwargs"] = kwargs
        return response

    monkeypatch.setattr(runtime.client.chat.completions, "create", _fake_create)

    first = runtime.observe()
    second = runtime.observe()
    context = runtime.build_context()

    assert first.current_subtask == "Pick up the center plate"
    assert second.current_subtask == "Pick up the center plate"
    assert recorded["calls"] == 1
    assert len(runtime.memory.entries) == 1
    assert "Working Memory" in context["working_memory_text"]
