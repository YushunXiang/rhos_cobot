"""Task decomposition: one-shot LLM call to split a prompt into subtasks."""

import dataclasses
import json
import logging
import re

from examples.piper_real.llm_utils import extract_message_json_text
from examples.piper_real.planner_config import PlannerConfig

_MAX_SUBTASKS = 10
_MAX_ATTEMPTS = 4

_VALID_TYPES = {"navigate", "manipulate"}
_ORDERED_SUBTASK_LINE_RE = re.compile(r"^\s*\d+\.\s+(?P<prompt>.+?)\s*$")
_LIGHT_STOPWORDS = {
    "a",
    "an",
    "and",
    "its",
    "of",
    "the",
    "to",
}

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


def _normalize_prompt_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def _significant_tokens(text: str) -> list[str]:
    return [
        token
        for token in _normalize_prompt_text(text).split()
        if token and token not in _LIGHT_STOPWORDS
    ]


def _is_subsequence(required_tokens: list[str], candidate_tokens: list[str]) -> bool:
    if not required_tokens:
        return False
    candidate_index = 0
    for required in required_tokens:
        while candidate_index < len(candidate_tokens) and candidate_tokens[candidate_index] != required:
            candidate_index += 1
        if candidate_index >= len(candidate_tokens):
            return False
        candidate_index += 1
    return True


def _parse_ordered_task_prompts(ordered_task_spec_text: str) -> list[str]:
    ordered_prompts: list[str] = []
    for line in ordered_task_spec_text.splitlines():
        match = _ORDERED_SUBTASK_LINE_RE.match(line)
        if not match:
            continue
        prompt = match.group("prompt").strip()
        if prompt:
            ordered_prompts.append(prompt)
    return ordered_prompts


def _canonicalize_prompt_to_ordered_spec(prompt: str, ordered_prompts: list[str]) -> str:
    normalized_prompt = _normalize_prompt_text(prompt)
    if not normalized_prompt:
        return prompt

    for ordered_prompt in ordered_prompts:
        if _normalize_prompt_text(ordered_prompt) == normalized_prompt:
            return ordered_prompt

    prompt_tokens = _significant_tokens(prompt)
    for ordered_prompt in ordered_prompts:
        ordered_tokens = _significant_tokens(ordered_prompt)
        if _is_subsequence(ordered_tokens, prompt_tokens):
            return ordered_prompt

    return prompt


def _canonicalize_subtasks_to_ordered_spec(
    subtasks: list["Subtask"], ordered_task_spec_text: str
) -> list["Subtask"]:
    ordered_prompts = _parse_ordered_task_prompts(ordered_task_spec_text)
    if not ordered_prompts:
        return subtasks

    canonicalized: list[Subtask] = []
    for subtask in subtasks:
        canonical_prompt = _canonicalize_prompt_to_ordered_spec(
            subtask.prompt, ordered_prompts
        )
        if canonical_prompt != subtask.prompt:
            logging.info(
                "Canonicalized decomposition prompt to ordered task spec: %r -> %r",
                subtask.prompt,
                canonical_prompt,
            )
        canonicalized.append(Subtask(type=subtask.type, prompt=canonical_prompt))
    return canonicalized


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

    def decompose(
        self,
        task_prompt: str,
        *,
        ordered_task_spec_text: str = "",
        working_memory_text: str = "",
        stage_estimate_text: str = "",
    ) -> list[Subtask]:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._attempt_decompose(
                    task_prompt,
                    ordered_task_spec_text=ordered_task_spec_text,
                    working_memory_text=working_memory_text,
                    stage_estimate_text=stage_estimate_text,
                )
            except (ValueError, KeyError, TypeError, self._openai.APIError) as exc:
                last_error = exc
                logging.warning(
                    "Decomposition attempt %d/%d failed: %s",
                    attempt + 1, _MAX_ATTEMPTS, exc,
                )
        raise DecompositionError(
            f"Task decomposition failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _attempt_decompose(
        self,
        task_prompt: str,
        *,
        ordered_task_spec_text: str = "",
        working_memory_text: str = "",
        stage_estimate_text: str = "",
    ) -> list[Subtask]:
        system_prompt = _SYSTEM_PROMPT
        if ordered_task_spec_text.strip():
            system_prompt += (
                " When an ordered task spec and working memory are provided, treat them as "
                "authoritative context. Stay aligned with the ordered task sequence, avoid "
                "re-introducing already completed prefix subtasks, and prefer decomposing the "
                "remaining suffix of the task. When a returned subtask matches an item in the "
                "ordered list, copy that ordered subtask text exactly instead of paraphrasing it. "
                "Do not add location qualifiers or state details that are not already present in "
                "the ordered list."
            )

        user_sections = [f"Task prompt:\n{task_prompt}"]
        if ordered_task_spec_text.strip():
            user_sections.append(f"Ordered task context:\n{ordered_task_spec_text}")
        if working_memory_text.strip():
            user_sections.append(f"Working memory:\n{working_memory_text}")
        if stage_estimate_text.strip():
            user_sections.append(f"Current stage estimate:\n{stage_estimate_text}")

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            max_tokens=self.config.task_decomposer_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.config.task_decomposer_enable_thinking,
                }
            },
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

        if ordered_task_spec_text.strip():
            result = _canonicalize_subtasks_to_ordered_spec(result, ordered_task_spec_text)

        logging.info(
            "Task decomposition: %s",
            json.dumps(
                {"subtasks": [{"type": s.type, "prompt": s.prompt} for s in result]},
                ensure_ascii=False,
            ),
        )
        return result
