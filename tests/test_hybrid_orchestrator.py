import dataclasses

from examples.piper_real import hybrid_orchestrator as orchestrator
from examples.piper_real import task_decomposer


class FakeEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def is_episode_complete(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class FakeTaskSpec:
    subtasks = ["stage one", "stage two"]
    done_index = 2

    def next_pending_label(self, completed_count: int) -> str:
        return self.subtasks[completed_count]


class FakeOrderedMemory:
    def __init__(self) -> None:
        self.task_spec = FakeTaskSpec()


@dataclasses.dataclass
class FakeBackend(orchestrator.HybridBackendBase):
    events: list[str] = dataclasses.field(default_factory=list)

    def execute_navigate(self, subtask, index: int, total: int):
        self.events.append(f"navigate:{index}/{total}:{subtask.prompt}")
        return orchestrator.HybridSubtaskResult(ok=True, completed=True)

    def before_manipulate(self, subtask, index: int, total: int) -> None:
        self.events.append(f"before-manipulate:{index}/{total}:{subtask.prompt}")

    def finalize(self) -> None:
        self.events.append("finalize")
        super().finalize()


def test_orchestrator_orders_batches_counts_and_marks_memory():
    env = FakeEnvironment()
    memory = FakeOrderedMemory()
    backend = FakeBackend(
        environment=env,
        policy_agent=object(),
        manipulation_planner=object(),
        ordered_task_memory_runtime=memory,
        ordered_reason_prefix="fake",
    )
    marks: list[tuple[int, str]] = []

    def build_subtasks(prompt: str, *, ordered_task_memory_runtime=None):
        assert ordered_task_memory_runtime is memory
        if prompt == "stage one":
            return [
                task_decomposer.Subtask(type="navigate", prompt="move to sink"),
                task_decomposer.Subtask(type="manipulate", prompt="pick plate"),
            ]
        if prompt == "stage two":
            return [
                task_decomposer.Subtask(type="manipulate", prompt="place plate"),
            ]
        raise AssertionError(prompt)

    def run_manipulation_subtask(*_args, subtask_prompt: str, **_kwargs):
        backend.events.append(f"manipulate:{subtask_prompt}")
        return {
            "executed_steps": 2 if subtask_prompt == "pick plate" else 3,
            "prompt_queries": 1,
            "completed": True,
            "stop_reason": "replanner_complete",
        }

    summary = orchestrator.run_hybrid_orchestrator(
        prompt="full task",
        backend=backend,
        build_subtask_list=build_subtasks,
        run_manipulation_subtask=run_manipulation_subtask,
        manipulation_config=orchestrator.HybridManipulationConfig(
            max_steps=8,
            replan_interval_steps=2,
            progress_complete_threshold=0.95,
            progress_stall_threshold=0.01,
            progress_stall_steps=3,
            progress_regression_threshold=0.05,
            progress_confirm_with_replanner=True,
            progress_head_mode="auto",
        ),
        mark_ordered_task_completed=lambda _runtime, idx, *, reason: marks.append(
            (idx, reason)
        ),
        get_ordered_completed_count=lambda _runtime, *, explicit_completed_count: explicit_completed_count,
    )

    assert backend.events == [
        "navigate:1/2:move to sink",
        "before-manipulate:1/2:pick plate",
        "manipulate:pick plate",
        "before-manipulate:2/2:place plate",
        "manipulate:place plate",
        "finalize",
    ]
    assert marks == [
        (0, "fake ordered subtask 1 completed"),
        (1, "fake ordered subtask 2 completed"),
    ]
    assert summary.total_subtasks == 2
    assert summary.navigate_subtasks == 1
    assert summary.manipulate_subtasks == 2
    assert summary.policy_steps == 5
    assert summary.prompt_queries == 2
    assert summary.status == "completed"
    assert env.closed is True
