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

        def close(self) -> None:
            recorded["closed"] = True

    class FakeTaskDecomposer:
        def __init__(self, _config) -> None:
            pass

        def decompose(self, _prompt: str):
            raise task_decomposer_mod.DecompositionError("empty subtasks")

    class FakeOfflineReplayNavigationPlanner:
        def __init__(self, _environment, _config) -> None:
            self.current_step = 3

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
