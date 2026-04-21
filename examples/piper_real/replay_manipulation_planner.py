"""Offline HDF5 replay adapter for VLM-driven manipulation prompt replanning."""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import re
from typing import Any

import cv2

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
    """Use replay frames to periodically refine the manipulation prompt for pi0.

    Works with any environment that exposes ``get_image(cam, idx)``, ``get_cursor()``,
    ``num_steps``, and ``camera_names`` — including ``PiperRealEnvironment`` on the
    real robot (duck-typed, not limited to offline replay).
    """

    def __init__(
        self,
        replay_environment: ReplayEnvironment,
        config: PlannerConfig,
        task_memory_runtime=None,
    ) -> None:
        from openai import OpenAI

        self.replay_environment = replay_environment
        self.config = config
        self.task_memory_runtime = task_memory_runtime
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
                    f"Active outer manipulation objective: {task_prompt}\n"
                    "Hard constraint: choose instructions only for this active objective. "
                    "Do not regress to earlier ordered subtasks, advance to later ordered "
                    "subtasks, or mark a different objective complete.\n"
                    "Manipulation objective: "
                    f"{task_prompt}\n"
                    f"Current policy prompt: {current_policy_prompt}\n"
                    f"Replay step: {step_idx}\n"
                    f"Policy steps executed in this manipulate subtask: {executed_policy_steps}\n"
                    f"Recent prompt history: {json.dumps(prompt_history[-5:], ensure_ascii=False)}"
                ),
            }
        ]
        if self.task_memory_runtime is not None:
            try:
                ordered_task_context = self.task_memory_runtime.build_context()
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Replay manipulation planner could not refresh ordered task memory: %s",
                    exc,
                )
            else:
                message_content.append(
                    {
                        "type": "text",
                        "text": (
                            "Ordered task context:\n"
                            f"{ordered_task_context['ordered_task_spec_text']}\n\n"
                            "Working memory:\n"
                            f"{ordered_task_context['working_memory_text']}\n\n"
                            "Current stage estimate:\n"
                            f"{ordered_task_context['stage_estimate_text']}\n\n"
                            "The ordered task context is diagnostic context only. "
                            "It must not override the active outer manipulation objective. "
                            "If the stage estimate appears behind the active objective, "
                            "continue the active objective instead of regressing."
                        ),
                    }
                )
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
            #response_format={"type": "json_object"},
            max_tokens=self.config.manipulation_replanner_max_tokens,
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
                        "Here, 'current manipulation objective' means the active outer objective "
                        "in the user message, not another ordered subtask mentioned in memory. "
                        "Do not ask the policy to navigate. "
                        "If ordered task context is provided, use it as scene memory while still "
                        "staying locked to the active outer objective."
                    ),
                },
                {
                    "role": "user",
                    "content": message_content,
                },
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.config.manipulation_replanner_enable_thinking,
                }
            },
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
        return self._enforce_active_objective(
            decision,
            task_prompt=task_prompt,
            current_policy_prompt=current_policy_prompt,
        )

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

    def _ordered_subtask_index(self, prompt: str) -> int | None:
        task_spec = getattr(self.task_memory_runtime, "task_spec", None)
        if task_spec is None:
            return None

        index = task_spec.subtask_index(prompt)
        if index < 0 or index >= getattr(task_spec, "done_index", index + 1):
            return None
        return index

    @staticmethod
    def _compact_command(text: str) -> str:
        text = re.sub(r"[^\w\s]+", " ", str(text).casefold())
        return re.sub(r"\s+", " ", text).strip()

    def _has_obvious_action_conflict(self, task_prompt: str, policy_prompt: str) -> bool:
        task = self._compact_command(task_prompt)
        prompt = self._compact_command(policy_prompt)
        opposites = (
            ("turn on", "turn off"),
            ("switch on", "switch off"),
            ("open", "close"),
            ("pick up", "put down"),
        )
        return any(left in task and right in prompt for left, right in opposites) or any(
            right in task and left in prompt for left, right in opposites
        )

    def _stage_confirms_target_complete(self, target_index: int) -> bool:
        task_memory_runtime = self.task_memory_runtime
        task_spec = getattr(task_memory_runtime, "task_spec", None)
        decision = getattr(task_memory_runtime, "_last_decision", None)
        if task_spec is None or decision is None:
            return True

        try:
            completed_count = len(task_spec.normalize_completed_prefix(decision.completed_subtasks))
            current_index = int(decision.current_subtask_index)
        except Exception:  # noqa: BLE001
            return False

        if completed_count > target_index:
            return True
        if current_index > target_index:
            return True
        return current_index >= task_spec.done_index

    def _enforce_active_objective(
        self,
        decision: ManipulationReplanDecision,
        *,
        task_prompt: str,
        current_policy_prompt: str,
    ) -> ManipulationReplanDecision:
        target_index = self._ordered_subtask_index(task_prompt)
        fallback_prompt = current_policy_prompt or task_prompt

        if decision.action == "complete":
            if target_index is not None and not self._stage_confirms_target_complete(target_index):
                logging.warning(
                    "Replay manipulate replanner completion rejected for active objective %r; "
                    "ordered stage has not confirmed it complete. Continuing with %r.",
                    task_prompt,
                    fallback_prompt,
                )
                return ManipulationReplanDecision(
                    action="continue",
                    prompt=fallback_prompt,
                    reason="completion rejected because ordered stage did not confirm the active objective",
                )
            return decision

        returned_index = self._ordered_subtask_index(decision.prompt)
        if target_index is not None and returned_index is not None and returned_index != target_index:
            logging.warning(
                "Replay manipulate replanner prompt rejected for active objective %r: "
                "returned prompt %r maps to ordered subtask %d, expected %d. Using %r.",
                task_prompt,
                decision.prompt,
                returned_index,
                target_index,
                task_prompt,
            )
            return ManipulationReplanDecision(
                action="continue",
                prompt=task_prompt,
                reason="prompt rejected because it targeted a different ordered subtask",
            )

        if self._has_obvious_action_conflict(task_prompt, decision.prompt):
            logging.warning(
                "Replay manipulate replanner prompt rejected for active objective %r: "
                "returned conflicting prompt %r. Using %r.",
                task_prompt,
                decision.prompt,
                task_prompt,
            )
            return ManipulationReplanDecision(
                action="continue",
                prompt=task_prompt,
                reason="prompt rejected because it conflicts with the active objective",
            )

        return decision

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
