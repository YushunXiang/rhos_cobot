"""Task decomposition: one-shot LLM call to split a prompt into subtasks."""

import dataclasses
import json
import logging

from examples.piper_real.llm_utils import extract_message_json_text
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
        import openai
        from openai import OpenAI

        self.config = config
        self._openai = openai
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def decompose(self, task_prompt: str) -> list[Subtask]:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._attempt_decompose(task_prompt)
            except (ValueError, KeyError, TypeError, self._openai.APIError) as exc:
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
        raw_text, raw_json = extract_message_json_text(response.choices[0].message)
        logging.debug("Task decomposer raw LLM response: %s", raw_text)
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
