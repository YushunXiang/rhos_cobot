"""Lightweight visual completion checks for known real-robot subtasks."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class VisualCompletionDecision:
    complete: bool
    reason: str
    metrics: dict[str, float]


def _compact_command(text: str) -> str:
    text = re.sub(r"[^\w\s]+", " ", str(text).casefold())
    return re.sub(r"\s+", " ", text).strip()


def _front_camera_name(environment: Any) -> str:
    preferred = getattr(environment, "front_camera_name", None)
    camera_names = tuple(getattr(environment, "camera_names", ()) or ())
    if preferred in camera_names:
        return str(preferred)
    for candidate in ("cam_high", "cam_front", "front", "camera_front"):
        if candidate in camera_names:
            return candidate
    if camera_names:
        return str(camera_names[0])
    return "cam_high"


def _central_workspace_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    y0 = int(height * 0.28)
    y1 = int(height * 0.92)
    x0 = int(width * 0.18)
    x1 = int(width * 0.82)
    return frame[y0:y1, x0:x1]


def _mask_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _largest_component_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return 0.0
    largest = max(int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels))
    return float(largest) / float(mask.size)


def _scene_metrics_for_hsv(hsv: np.ndarray) -> dict[str, float]:
    # Bread in the deployment setup is light tan/yellow. The thresholds are
    # deliberately conservative and require a connected region, not just noise.
    bread_mask = cv2.inRange(hsv, np.array([8, 25, 80]), np.array([45, 190, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 35, 45]), np.array([95, 255, 255]))
    plate_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 75, 255]))

    kernel = np.ones((5, 5), np.uint8)
    bread_mask = cv2.morphologyEx(bread_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, kernel)

    return {
        "bread_ratio": _mask_ratio(bread_mask),
        "bread_largest_ratio": _largest_component_ratio(bread_mask),
        "green_ratio": _mask_ratio(green_mask),
        "green_largest_ratio": _largest_component_ratio(green_mask),
        "plate_ratio": _mask_ratio(plate_mask),
        "plate_largest_ratio": _largest_component_ratio(plate_mask),
    }


def _scene_metrics(frame: np.ndarray) -> dict[str, float]:
    roi = _central_workspace_roi(frame)
    rgb_metrics = _scene_metrics_for_hsv(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV))
    bgr_metrics = _scene_metrics_for_hsv(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    return {
        key: max(rgb_metrics[key], bgr_metrics[key])
        for key in rgb_metrics
    }


class SandwichVisualCompletionDetector:
    """Detect completion for sandwich assembly subtasks from camera frames."""

    def __init__(self, task_prompt: str, *, required_hits: int = 2) -> None:
        self.task_prompt = task_prompt
        self.required_hits = max(1, int(required_hits))
        self._hits = 0
        compact = _compact_command(task_prompt)
        if "second" in compact and "bread" in compact:
            self.kind = ""
        elif "lettuce" in compact:
            self.kind = "lettuce"
        elif "bread" in compact and "first" in compact:
            self.kind = "bread"
        else:
            self.kind = ""

    @property
    def enabled(self) -> bool:
        return bool(self.kind)

    def observe(self, environment: Any) -> VisualCompletionDecision:
        if not self.enabled:
            return VisualCompletionDecision(False, "unsupported visual subtask", {})

        cam_name = _front_camera_name(environment)
        cursor = environment.get_cursor() if hasattr(environment, "get_cursor") else 0
        frame = environment.get_image(cam_name, max(0, int(cursor) - 1))
        metrics = _scene_metrics(frame)

        if self.kind == "bread":
            instant_complete = (
                metrics["bread_ratio"] >= 0.018
                and metrics["bread_largest_ratio"] >= 0.010
                and metrics["plate_ratio"] >= 0.020
            )
            reason = (
                "bread-like connected region is visible in the plate workspace"
            )
        else:
            instant_complete = (
                metrics["green_ratio"] >= 0.012
                and metrics["green_largest_ratio"] >= 0.006
                and metrics["bread_ratio"] >= 0.010
            )
            reason = (
                "lettuce-like connected region is visible on the bread workspace"
            )

        if instant_complete:
            self._hits += 1
        else:
            self._hits = 0

        complete = self._hits >= self.required_hits
        return VisualCompletionDecision(
            complete=complete,
            reason=f"{reason}; stable_hits={self._hits}/{self.required_hits}",
            metrics=metrics,
        )
