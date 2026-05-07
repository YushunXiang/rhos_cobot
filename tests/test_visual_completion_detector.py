from types import SimpleNamespace

import numpy as np


def _sandwich_frame(*, bread: bool = False, lettuce: bool = False) -> np.ndarray:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :] = (70, 70, 70)
    frame[80:180, 62:162] = (235, 235, 235)
    if bread:
        frame[104:152, 84:140] = (194, 150, 82)
    if lettuce:
        frame[112:144, 92:148] = (55, 150, 55)
    return frame


class _FakeImageEnvironment:
    camera_names = ("cam_high",)
    front_camera_name = "cam_high"

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def get_cursor(self) -> int:
        return 1

    def get_image(self, _cam_name: str, _idx: int) -> np.ndarray:
        return self.frame


def test_sandwich_visual_completion_detector_requires_stable_bread_hits():
    from examples.piper_real.visual_completion_detector import (
        SandwichVisualCompletionDetector,
    )

    detector = SandwichVisualCompletionDetector(
        "Place the first slice of bread on the plate.",
        required_hits=2,
    )
    env = _FakeImageEnvironment(_sandwich_frame(bread=True))

    first = detector.observe(env)
    second = detector.observe(env)

    assert first.complete is False
    assert second.complete is True
    assert second.metrics["bread_ratio"] > 0.018
    assert second.metrics["plate_ratio"] > 0.020


def test_sandwich_visual_completion_detector_detects_lettuce_on_bread():
    from examples.piper_real.visual_completion_detector import (
        SandwichVisualCompletionDetector,
    )

    detector = SandwichVisualCompletionDetector(
        "Place the lettuce leaf on the first slice of bread.",
        required_hits=1,
    )

    decision = detector.observe(
        _FakeImageEnvironment(_sandwich_frame(bread=True, lettuce=True))
    )

    assert decision.complete is True
    assert decision.metrics["green_ratio"] > 0.012
    assert decision.metrics["bread_ratio"] > 0.010


def test_sandwich_visual_completion_detector_ignores_second_bread_prompt():
    from examples.piper_real.visual_completion_detector import (
        SandwichVisualCompletionDetector,
    )

    detector = SandwichVisualCompletionDetector(
        "Place the second slice of bread on the lettuce leaf.",
    )

    assert detector.enabled is False


def test_run_manipulation_subtask_can_complete_from_visual_detector():
    from examples.piper_real import main as main_module

    class FakeEnvironment:
        camera_names = ("cam_high",)
        front_camera_name = "cam_high"

        def __init__(self) -> None:
            self._cursor = 0
            self.prompts = []

        def is_episode_complete(self) -> bool:
            return False

        def get_cursor(self) -> int:
            return self._cursor

        def get_observation(self) -> dict:
            self._cursor += 1
            return {"step": self._cursor}

        def apply_action(self, _action: dict) -> None:
            pass

        def get_image(self, _cam_name: str, _idx: int) -> np.ndarray:
            return _sandwich_frame(bread=True)

        def set_prompt(self, prompt: str) -> None:
            self.prompts.append(prompt)

    class FakeAgent:
        policy_metadata = {"has_progress_head": False}

        def reset(self) -> None:
            pass

        def get_action(self, _observation: dict) -> dict:
            return {"actions": np.zeros(14, dtype=np.float32)}

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                action="continue",
                prompt=kwargs["current_policy_prompt"],
                reason="not complete",
            )

    planner = FakePlanner()
    result = main_module._run_manipulation_subtask(
        FakeEnvironment(),
        FakeAgent(),
        planner,
        subtask_prompt="Place the first slice of bread on the plate.",
        max_steps=6,
        replan_interval_steps=2,
        progress_complete_threshold=0.85,
        progress_stall_threshold=0.02,
        progress_stall_steps=3,
        progress_regression_threshold=0.1,
        progress_confirm_with_replanner=False,
    )

    assert result["completed"] is True
    assert result["completed_by_visual"] is True
    assert result["stop_reason"] == "visual_complete"
    assert result["executed_steps"] == 4
    assert planner.calls == 2
