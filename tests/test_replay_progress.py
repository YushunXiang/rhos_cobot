import numpy as np
from types import SimpleNamespace


def test_progress_tracker_marks_complete():
    from examples.piper_real.main import ReplayTaskProgressTracker

    tracker = ReplayTaskProgressTracker(
        complete_threshold=0.85,
        stall_threshold=0.02,
        stall_steps=3,
        regression_threshold=0.1,
    )

    decision = tracker.observe(0.9)

    assert decision.event == "complete"


def test_progress_tracker_marks_stall_after_small_deltas():
    from examples.piper_real.main import ReplayTaskProgressTracker

    tracker = ReplayTaskProgressTracker(
        complete_threshold=0.95,
        stall_threshold=0.02,
        stall_steps=3,
        regression_threshold=0.1,
    )

    assert tracker.observe(0.10).event == "continue"
    assert tracker.observe(0.11).event == "continue"
    decision = tracker.observe(0.115)

    assert decision.event == "stall"


def test_progress_tracker_marks_regression_after_backslide():
    from examples.piper_real.main import ReplayTaskProgressTracker

    tracker = ReplayTaskProgressTracker(
        complete_threshold=0.95,
        stall_threshold=0.02,
        stall_steps=3,
        regression_threshold=0.1,
    )

    assert tracker.observe(0.7).event == "continue"
    decision = tracker.observe(0.55)

    assert decision.event == "regression"


def test_replay_manipulation_subtask_uses_progress_before_replanner():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        def __init__(self) -> None:
            self._cursor = 0

        def is_episode_complete(self) -> bool:
            return False

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, action: dict) -> None:
            assert np.isclose(action["progress"], 0.9)

        def set_prompt(self, _prompt: str) -> None:
            raise AssertionError("progress-first completion should not update prompts")

    class FakeAgent:
        def __init__(self) -> None:
            self.policy_metadata = {"has_progress_head": True}

        def reset(self) -> None:
            pass

        def get_action(self, _observation: dict) -> dict:
            return {
                "actions": np.zeros(14, dtype=np.float32),
                "progress": np.float32(0.9),
            }

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(action="continue", prompt="unused", reason="unused")

    planner = FakePlanner()
    result = main_module._run_manipulation_subtask(
        FakeEnvironment(),
        FakeAgent(),
        planner,
        subtask_prompt="pick plate",
        max_steps=4,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
    )

    assert result["completed_by_progress"] is True
    assert result["completed_by_replan"] is False
    assert planner.calls == []


def test_replay_manipulation_subtask_force_enables_progress_without_metadata():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        def __init__(self) -> None:
            self._cursor = 0

        def is_episode_complete(self) -> bool:
            return False

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, action: dict) -> None:
            assert np.isclose(action["progress"], 0.9)

        def set_prompt(self, _prompt: str) -> None:
            raise AssertionError("forced progress completion should not update prompts")

    class FakeAgent:
        policy_metadata = {"has_progress_head": False}

        def reset(self) -> None:
            pass

        def get_action(self, _observation: dict) -> dict:
            return {
                "actions": np.zeros(14, dtype=np.float32),
                "progress": np.float32(0.9),
            }

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(action="continue", prompt="unused", reason="unused")

    planner = FakePlanner()
    result = main_module._run_manipulation_subtask(
        FakeEnvironment(),
        FakeAgent(),
        planner,
        subtask_prompt="pick plate",
        max_steps=4,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
        progress_head_mode="force",
    )

    assert result["completed_by_progress"] is True
    assert result["completed_by_replan"] is False
    assert planner.calls == []


def test_replay_manipulation_subtask_off_disables_progress_metadata():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        def __init__(self) -> None:
            self._cursor = 0
            self.prompts = []

        def is_episode_complete(self) -> bool:
            return self._cursor >= 3

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, _action: dict) -> None:
            pass

        def set_prompt(self, prompt: str) -> None:
            self.prompts.append(prompt)

    class FakeAgent:
        policy_metadata = {"has_progress_head": True}

        def reset(self) -> None:
            pass

        def get_action(self, _observation: dict) -> dict:
            return {
                "actions": np.zeros(14, dtype=np.float32),
                "progress": np.float32(0.9),
            }

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["executed_policy_steps"] == 0:
                return SimpleNamespace(action="continue", prompt="keep going", reason="start")
            return SimpleNamespace(action="complete", prompt="", reason="done")

    planner = FakePlanner()
    result = main_module._run_manipulation_subtask(
        FakeEnvironment(),
        FakeAgent(),
        planner,
        subtask_prompt="turn tap",
        max_steps=4,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
        progress_head_mode="off",
    )

    assert result["completed_by_progress"] is False
    assert result["completed_by_replan"] is True
    assert [call["executed_policy_steps"] for call in planner.calls] == [0, 2]


def test_replay_manipulation_subtask_falls_back_to_replanner_without_progress_head():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        def __init__(self) -> None:
            self._cursor = 0
            self.prompts = []

        def is_episode_complete(self) -> bool:
            return self._cursor >= 3

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, _action: dict) -> None:
            pass

        def set_prompt(self, prompt: str) -> None:
            self.prompts.append(prompt)

    class FakeAgent:
        def __init__(self) -> None:
            self.policy_metadata = {"has_progress_head": False}
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def get_action(self, _observation: dict) -> dict:
            return {"actions": np.zeros(14, dtype=np.float32)}

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["executed_policy_steps"] == 0:
                return SimpleNamespace(action="continue", prompt="grasp handle", reason="keep going")
            return SimpleNamespace(action="complete", prompt="", reason="done")

    env = FakeEnvironment()
    agent = FakeAgent()
    planner = FakePlanner()
    result = main_module._run_manipulation_subtask(
        env,
        agent,
        planner,
        subtask_prompt="turn tap",
        max_steps=4,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
    )

    assert result["completed_by_progress"] is False
    assert result["completed_by_replan"] is True
    assert result["completed"] is True
    assert [call["executed_policy_steps"] for call in planner.calls] == [0, 2]
    assert env.prompts == ["grasp handle"]


def test_replay_manipulation_subtask_reports_step_cap_when_incomplete():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        def __init__(self) -> None:
            self._cursor = 0

        def is_episode_complete(self) -> bool:
            return False

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, _action: dict) -> None:
            pass

        def set_prompt(self, _prompt: str) -> None:
            pass

    class FakeAgent:
        def __init__(self) -> None:
            self.policy_metadata = {"has_progress_head": False}

        def reset(self) -> None:
            pass

        def get_action(self, _observation: dict) -> dict:
            return {"actions": np.zeros(14, dtype=np.float32)}

    class FakePlanner:
        def plan(self, **kwargs):
            return SimpleNamespace(action="continue", prompt="keep trying", reason="not done")

    result = main_module._run_manipulation_subtask(
        FakeEnvironment(),
        FakeAgent(),
        FakePlanner(),
        subtask_prompt="turn tap",
        max_steps=4,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
    )

    assert result["completed"] is False
    assert result["completed_by_progress"] is False
    assert result["completed_by_replan"] is False
    assert result["stop_reason"] == "step_cap"
