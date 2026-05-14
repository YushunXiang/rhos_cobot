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


def test_real_backend_streams_visualizer_context_updates_and_close():
    class LiveEnvironment(FakeEnvironment):
        ros_operator = object()
        num_steps = 100

        def __init__(self) -> None:
            super().__init__()
            self.cursor = 7

        def get_cursor(self) -> int:
            return self.cursor

        def set_prompt(self, prompt: str) -> None:
            events.append(f"set-prompt:{prompt}")

        def refresh_observation_cache(self) -> None:
            events.append("refresh-cache")

    class FakeVisualizer:
        def __init__(self) -> None:
            self.contexts: list[tuple[int, int, str, str]] = []
            self.updates: list[tuple[int, str]] = []
            self.closed = False

        def set_subtask_context(
            self,
            subtask_idx: int,
            total_subtasks: int,
            subtask_type: str,
            subtask_prompt: str,
        ) -> None:
            self.contexts.append(
                (subtask_idx, total_subtasks, subtask_type, subtask_prompt)
            )

        def update(self, step: int, *, extra_info: str = "") -> bool:
            self.updates.append((step, extra_info))
            return True

        def close(self) -> None:
            self.closed = True

    events: list[str] = []
    visualizer = FakeVisualizer()
    env = LiveEnvironment()

    def navigate_func(
        prompt,
        ros_operator,
        *,
        dry_run=False,
        frame_tick_callback=None,
    ):
        events.append(f"navigate:{prompt}:{dry_run}:{ros_operator is env.ros_operator}")
        assert frame_tick_callback is not None
        frame_tick_callback()
        return type(
            "NavigationResult",
            (),
            {
                "ok": True,
                "prompt": prompt,
                "routine_name": "default_demo",
                "executed_steps": 1,
                "error": "",
            },
        )()

    backend = orchestrator.RealHybridBackend(
        environment=env,
        policy_agent=object(),
        manipulation_planner=object(),
        visualizer=visualizer,
        navigate_func=navigate_func,
        refresh_observation_cache=(
            lambda environment, *, context: environment.refresh_observation_cache()
        ),
    )

    def build_subtasks(_prompt: str, *, ordered_task_memory_runtime=None):
        assert ordered_task_memory_runtime is None
        return [
            task_decomposer.Subtask(type="navigate", prompt="move to table"),
            task_decomposer.Subtask(type="manipulate", prompt="pick cup"),
        ]

    def run_manipulation_subtask(
        *_args,
        subtask_prompt: str,
        visualizer=None,
        **_kwargs,
    ):
        events.append(f"manipulate:{subtask_prompt}:{visualizer is visualizer_arg}")
        return {"executed_steps": 2, "prompt_queries": 1, "completed": True}

    visualizer_arg = visualizer
    summary = orchestrator.run_hybrid_orchestrator(
        prompt="move and pick",
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
    )

    assert visualizer.contexts == [
        (1, 2, "navigate", "move to table"),
        (2, 2, "manipulate", "pick cup"),
    ]
    assert visualizer.updates == [(7, "navigate subtask 1/2")]
    assert visualizer.closed is True
    assert summary.status == "completed"
    assert events == [
        "navigate:move to table:True:True",
        "set-prompt:pick cup",
        "refresh-cache",
        "manipulate:pick cup:True",
    ]
