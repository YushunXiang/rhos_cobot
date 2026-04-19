"""Offline HDF5 replay adapter for VLM-driven manipulation prompt replanning."""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Any

import cv2
from openai import OpenAI

from examples.piper_real.llm_utils import extract_message_json_text
from examples.piper_real.planner_config import PlannerConfig
from examples.piper_real.replay_env import ReplayEnvironment


class ManipulationPromptPlannerError(RuntimeError):
    """Raised when the manipulation prompt replanner returns an unusable response."""


@dataclasses.dataclass
class ManipulationReplanDecision:
    action: str
    prompt: str = ""
    reason: str = ""


class ReplayManipulationPromptPlanner:
    """Use replay frames to periodically refine the manipulation prompt for pi0."""

    def __init__(self, replay_environment: ReplayEnvironment, config: PlannerConfig) -> None:
        self.replay_environment = replay_environment
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def plan(
        self,
        *,
        task_prompt: str,
        current_policy_prompt: str,
        executed_policy_steps: int,
        prompt_history: list[dict[str, Any]],
    ) -> ManipulationReplanDecision:
        if self.replay_environment.num_steps == 0:
            raise ManipulationPromptPlannerError("replay dataset is empty")

        step_idx = self._current_frame_index()
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Manipulation objective: "
                    f"{task_prompt}\n"
                    f"Current policy prompt: {current_policy_prompt}\n"
                    f"Replay step: {step_idx}\n"
                    f"Policy steps executed in this manipulate subtask: {executed_policy_steps}\n"
                    f"Recent prompt history: {json.dumps(prompt_history[-5:], ensure_ascii=False)}"
                ),
            }
        ]
        for cam_name in self.replay_environment.camera_names:
            message_content.append(
                {
                    "type": "text",
                    "text": f"Camera view: {cam_name}",
                }
            )
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(cam_name, step_idx)},
                }
            )

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a manipulation-stage replanner for a mobile manipulation robot. "
                        "Your job is to choose the next short language instruction for the arm policy. "
                        "Return JSON only with no markdown, no prose, and no thinking process. "
                        "Allowed outputs are: "
                        "{\"action\":\"continue\",\"prompt\":str,\"reason\":str} "
                        "or {\"action\":\"complete\",\"reason\":str}. "
                        "Use 'continue' when the robot should keep manipulating and provide a concise, "
                        "single-stage policy prompt. "
                        "Use 'complete' only when the current manipulation objective is already finished "
                        "and control should return to the outer long-horizon planner. "
                        "Do not ask the policy to navigate."
                    ),
                },
                {
                    "role": "user",
                    "content": message_content,
                },
            ],
        )
        try:
            raw_text, raw_json = extract_message_json_text(response.choices[0].message)
        except ValueError as exc:
            raise ManipulationPromptPlannerError(str(exc)) from exc

        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ManipulationPromptPlannerError(
                f"manipulation replanner returned invalid JSON: {raw_json[:400]}"
            ) from exc
        if not isinstance(payload, dict):
            raise ManipulationPromptPlannerError("manipulation replanner response must be a JSON object")
        decision = self._normalize_decision(payload)
        decision.reason = decision.reason or raw_text
        return decision

    def _current_frame_index(self) -> int:
        cursor = self.replay_environment.get_cursor()
        if cursor >= self.replay_environment.num_steps:
            return self.replay_environment.num_steps - 1
        return cursor

    def _encode_image(self, cam_name: str, step_idx: int) -> str:
        frame = self.replay_environment.get_image(cam_name, step_idx)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ManipulationPromptPlannerError(f"failed to encode replay frame {step_idx} from {cam_name}")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

    def _normalize_decision(self, payload: dict[str, Any]) -> ManipulationReplanDecision:
        action = str(payload.get("action", "")).strip().lower()
        if action == "complete":
            return ManipulationReplanDecision(
                action="complete",
                reason=str(payload.get("reason", "manipulation objective complete")).strip(),
            )
        if action != "continue":
            raise ManipulationPromptPlannerError(f"unsupported manipulation replanner action: {action!r}")

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ManipulationPromptPlannerError("continue response must include a non-empty prompt")
        return ManipulationReplanDecision(
            action="continue",
            prompt=prompt,
            reason=str(payload.get("reason", "")).strip(),
        )
