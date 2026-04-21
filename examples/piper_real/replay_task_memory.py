"""Ordered task-spec stage analysis and rolling memory for replay hybrid mode."""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2

from examples.piper_real.llm_utils import extract_message_json_text
from examples.piper_real.planner_config import PlannerConfig
from examples.piper_real.replay_env import ReplayEnvironment


def _compact_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_label(label: str | None) -> str:
    text = _compact_text(label).casefold()
    text = re.sub(r"[^\w\s]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _round_values(values: Any, digits: int = 4) -> list[float] | None:
    if values is None:
        return None
    try:
        return [round(float(value), digits) for value in values]
    except TypeError:
        return [round(float(values), digits)]


@dataclasses.dataclass(frozen=True)
class OrderedTaskSpec:
    name: str
    total_task: str
    subtasks: list[str]
    done_label: str = "task complete"
    uncertain_label: str = "uncertain"

    def __post_init__(self) -> None:
        cleaned = [_compact_text(item) for item in self.subtasks if _compact_text(item)]
        if not cleaned:
            raise ValueError("OrderedTaskSpec requires at least one subtask.")
        if len({_normalize_label(item) for item in cleaned}) != len(cleaned):
            raise ValueError(f"OrderedTaskSpec subtasks must be unique: {cleaned}")
        object.__setattr__(self, "subtasks", cleaned)
        object.__setattr__(self, "total_task", _compact_text(self.total_task))
        object.__setattr__(self, "done_label", _compact_text(self.done_label) or "task complete")
        object.__setattr__(self, "uncertain_label", _compact_text(self.uncertain_label) or "uncertain")

    @property
    def done_index(self) -> int:
        return len(self.subtasks)

    @property
    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(self.subtasks + [self.done_label, self.uncertain_label])

    def _canonical_label(self, label: str | None) -> str | None:
        normalized = _normalize_label(label)
        for candidate in self.allowed_labels:
            if _normalize_label(candidate) == normalized:
                return candidate
        return None

    def coerce_subtask(self, label: str | None) -> str | None:
        return self._canonical_label(label)

    def subtask_index(self, label: str | None) -> int:
        normalized = self.coerce_subtask(label)
        if normalized is None:
            return -1
        if normalized == self.done_label:
            return self.done_index
        if normalized == self.uncertain_label:
            return -1
        return self.subtasks.index(normalized)

    def label_from_index(self, index: int) -> str:
        if index == self.done_index:
            return self.done_label
        if 0 <= index < len(self.subtasks):
            return self.subtasks[index]
        return self.uncertain_label

    def prefix(self, completed_count: int) -> list[str]:
        completed_count = max(0, min(int(completed_count), len(self.subtasks)))
        return list(self.subtasks[:completed_count])

    def normalize_completed_prefix(self, values: list[str] | tuple[str, ...] | None) -> list[str]:
        if values is None:
            return []
        requested = {_normalize_label(item) for item in values if _compact_text(item)}
        normalized: list[str] = []
        for label in self.subtasks:
            if _normalize_label(label) in requested:
                normalized.append(label)
            else:
                break
        return normalized

    def next_pending_label(self, completed_count: int) -> str:
        completed_count = max(0, min(int(completed_count), len(self.subtasks)))
        if completed_count >= len(self.subtasks):
            return self.done_label
        return self.subtasks[completed_count]

    def next_after_current(self, current_index: int) -> str:
        if current_index < 0:
            return self.next_pending_label(0)
        if current_index + 1 >= len(self.subtasks):
            return self.done_label
        return self.subtasks[current_index + 1]

    def ordered_subtasks_text(self) -> str:
        return "\n".join(
            f"{index + 1}. {subtask}" for index, subtask in enumerate(self.subtasks)
        )

    def as_prompt_text(self) -> str:
        return (
            f"Overall task: {self.total_task}\n"
            "Ordered subtask list:\n"
            f"{self.ordered_subtasks_text()}"
        )


def load_ordered_task_spec(path: str | Path) -> OrderedTaskSpec:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"Task spec not found: {spec_path}")

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task spec must be a JSON object: {spec_path}")

    return OrderedTaskSpec(
        name=str(payload.get("name") or spec_path.stem),
        total_task=str(payload.get("total_task") or "").strip(),
        subtasks=list(payload.get("subtasks") or []),
        done_label=str(payload.get("done_label") or "task complete").strip() or "task complete",
        uncertain_label=str(payload.get("uncertain_label") or "uncertain").strip() or "uncertain",
    )


@dataclasses.dataclass
class TaskMemoryEntry:
    step_index: int
    current_subtask: str
    current_subtask_index: int
    completed_subtasks: list[str]
    next_subtask: str
    confidence: float
    evidence: str
    memory_update: str
    state_summary: str


class RollingTaskMemory:
    def __init__(self, max_entries: int = 12) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.entries: list[TaskMemoryEntry] = []

    def add(self, entry: TaskMemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def highest_completed_count(self, task_spec: OrderedTaskSpec) -> int:
        highest = 0
        for entry in self.entries:
            highest = max(
                highest,
                len(task_spec.normalize_completed_prefix(entry.completed_subtasks)),
            )
        return highest

    def as_prompt_text(self, task_spec: OrderedTaskSpec) -> str:
        if not self.entries:
            return "No prior working memory."

        last = self.entries[-1]
        completed_count = len(task_spec.normalize_completed_prefix(last.completed_subtasks))
        total_count = len(task_spec.subtasks)
        progress_pct = (completed_count / total_count) * 100 if total_count > 0 else 0.0

        lines = ["### Working Memory ###"]
        lines.append(f"Task progress: {completed_count}/{total_count} ({progress_pct:.0f}%)")
        lines.append(f"Current/next candidate: {task_spec.next_pending_label(completed_count)}")

        completed_prefix = task_spec.prefix(self.highest_completed_count(task_spec))
        if completed_prefix:
            lines.append("Completed subtasks:")
            for index, label in enumerate(completed_prefix, start=1):
                lines.append(f"  {index}. [DONE] {label}")
        else:
            lines.append("Completed subtasks: (not started)")

        all_facts: list[str] = []
        seen: set[str] = set()
        for entry in self.entries:
            for raw in (entry.state_summary, entry.memory_update):
                fact = _compact_text(raw)
                if len(fact) <= 4 or fact in seen:
                    continue
                seen.add(fact)
                all_facts.append(fact)

        if all_facts:
            lines.append("Key object and environment facts:")
            initial_facts = all_facts[:3]
            for fact in initial_facts:
                lines.append(f"  - [core] {fact}")
            if len(all_facts) > 8:
                lines.append("  - ...")
                recent_facts = all_facts[-5:]
            elif len(all_facts) > 3:
                recent_facts = all_facts[3:]
            else:
                recent_facts = []
            for fact in recent_facts:
                lines.append(f"  - [recent] {fact}")

        lines.append("Most recent stage estimates:")
        for entry in self.entries[-3:]:
            lines.append(
                f"  - Step {entry.step_index}: {entry.current_subtask} "
                f"(confidence {entry.confidence:.2f})"
            )
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class TaskStageDecision:
    current_subtask: str
    current_subtask_index: int
    completed_subtasks: list[str]
    next_subtask: str
    confidence: float
    evidence: str
    memory_update: str
    state_summary: str
    sequence_enforced: bool = False

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "current_subtask": self.current_subtask,
            "current_subtask_index": self.current_subtask_index,
            "completed_subtasks": list(self.completed_subtasks),
            "next_subtask": self.next_subtask,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "memory_update": self.memory_update,
            "state_summary": self.state_summary,
            "sequence_enforced": self.sequence_enforced,
        }


class TaskStageAnalysisError(RuntimeError):
    """Raised when ordered task-stage analysis returns an unusable response."""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict):
                        return payload
        start = cleaned.find("{", start + 1)
    raise ValueError(f"Could not parse JSON object from response: {text}")


def _extract_last_match(text: str, patterns: list[str]) -> str | None:
    last_match = None
    last_pos = (-1, -1)
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            for group in match.groups():
                if group is None:
                    continue
                match_pos = (match.start(), match.end())
                if match_pos >= last_pos:
                    last_match = group.strip()
                    last_pos = match_pos
    return last_match


def _extract_list_from_text(text: str, field_name: str) -> list[str]:
    match = re.search(
        rf"{field_name}[^\[]*\[(.*?)\]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    inner = match.group(1)
    return [item for item in re.findall(r'["“]([^"\n\r”]+)["”]', inner) if item.strip()]


def _allowed_label_pattern(task_spec: OrderedTaskSpec) -> str:
    return "|".join(
        sorted(
            (re.escape(label) for label in task_spec.allowed_labels),
            key=len,
            reverse=True,
        )
    )


def _extract_decision_from_text(text: str, task_spec: OrderedTaskSpec) -> dict[str, Any]:
    decision: dict[str, Any] = {}
    label_pattern = _allowed_label_pattern(task_spec)

    current_subtask = _extract_last_match(
        text,
        [
            rf"current_subtask[^\n\r]{{0,40}}?[:=]\s*[\"“]?({label_pattern})",
            rf"(?:current task|active subtask)[^\n\r]{{0,30}}?(?:is|=)\s*[\"“]?({label_pattern})",
        ],
    )
    if current_subtask is not None:
        decision["current_subtask"] = current_subtask

    next_subtask = _extract_last_match(
        text,
        [
            rf"next_subtask[^\n\r]{{0,40}}?[:=]\s*[\"“]?({label_pattern})",
            rf"(?:next subtask|next task)[^\n\r]{{0,30}}?(?:is|=)\s*[\"“]?({label_pattern})",
        ],
    )
    if next_subtask is not None:
        decision["next_subtask"] = next_subtask

    completed_subtasks = _extract_list_from_text(text, "completed_subtasks")
    if completed_subtasks:
        decision["completed_subtasks"] = completed_subtasks

    current_index = _extract_last_match(
        text,
        [r"current_subtask_index[^\n\r]{0,20}?[:=]\s*(-?\d+)"],
    )
    if current_index is not None:
        try:
            decision["current_subtask_index"] = int(current_index)
        except ValueError:
            pass

    confidence = _extract_last_match(
        text,
        [r"confidence[^\n\r]{0,20}?[:=]\s*(1(?:\.0+)?|0(?:\.\d+)?)"],
    )
    if confidence is not None:
        try:
            decision["confidence"] = float(confidence)
        except ValueError:
            pass

    for field in ("evidence", "memory_update", "state_summary"):
        value = _extract_last_match(
            text,
            [
                rf"{field}[^\n\r]{{0,20}}?[\"“]([^\"\n\r”]+)[\"”]",
                rf"{field}[^\n\r]{{0,20}}?[:=]\s*[\"“]?([^\"\n\r”]+)",
            ],
        )
        if value:
            decision[field] = value

    if "current_subtask" not in decision:
        fallback_label = _extract_last_match(text, [rf"({label_pattern})"])
        if fallback_label is not None:
            decision["current_subtask"] = fallback_label

    return decision


def _coerce_completed_subtasks(raw: Any, task_spec: OrderedTaskSpec) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return task_spec.normalize_completed_prefix([str(item).strip() for item in raw])
    if isinstance(raw, str):
        pieces = re.split(r"[,\n;]+", raw)
        return task_spec.normalize_completed_prefix([piece.strip() for piece in pieces])
    return []


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_decision(
    raw: dict[str, Any],
    raw_response: str,
    task_spec: OrderedTaskSpec,
    memory: RollingTaskMemory,
) -> dict[str, Any]:
    raw_completed = _coerce_completed_subtasks(raw.get("completed_subtasks"), task_spec)
    memory_completed_count = memory.highest_completed_count(task_spec)

    current_subtask = task_spec.coerce_subtask(raw.get("current_subtask"))
    current_index = raw.get("current_subtask_index")
    try:
        current_index = int(current_index)
    except (TypeError, ValueError):
        current_index = task_spec.subtask_index(current_subtask)

    confidence = _coerce_confidence(
        raw.get("confidence"),
        default=0.25 if raw_response.strip() else 0.0,
    )

    if current_subtask == task_spec.done_label or current_index >= task_spec.done_index:
        normalized_subtask = task_spec.done_label
        normalized_index = task_spec.done_index
    elif current_subtask in task_spec.subtasks:
        normalized_index = task_spec.subtask_index(current_subtask)
        normalized_subtask = current_subtask
    elif 0 <= current_index < task_spec.done_index:
        normalized_index = current_index
        normalized_subtask = task_spec.label_from_index(current_index)
    else:
        normalized_index = memory_completed_count
        normalized_subtask = task_spec.label_from_index(normalized_index)

    if normalized_subtask != task_spec.done_label:
        strict_threshold = 0.5
        relaxed_threshold = 0.35
        is_skipping = normalized_index > memory_completed_count
        effective_threshold = strict_threshold if is_skipping else relaxed_threshold

        if confidence >= effective_threshold:
            normalized_index = max(normalized_index, len(raw_completed), memory_completed_count)
        else:
            normalized_index = memory_completed_count

        if normalized_index >= task_spec.done_index:
            normalized_subtask = task_spec.done_label
            normalized_index = task_spec.done_index
        else:
            normalized_subtask = task_spec.label_from_index(normalized_index)

    if normalized_subtask == task_spec.done_label:
        completed_subtasks = list(task_spec.subtasks)
        next_subtask = task_spec.done_label
    else:
        completed_subtasks = task_spec.prefix(normalized_index)
        next_subtask = task_spec.next_after_current(normalized_index)

    evidence = _compact_text(raw.get("evidence"))
    if not evidence:
        evidence = (
            f"Based on the current replay images, robot state, and memory, "
            f"the robot appears to be executing '{normalized_subtask}'."
        )

    memory_update = _compact_text(raw.get("memory_update"))
    if not memory_update:
        if normalized_subtask == task_spec.done_label:
            memory_update = "The overall task appears complete."
        else:
            memory_update = (
                f"The robot has advanced to subtask {normalized_index + 1}: "
                f"{normalized_subtask}."
            )

    state_summary = _compact_text(raw.get("state_summary"))
    if not state_summary:
        state_summary = evidence

    return {
        "current_subtask": normalized_subtask,
        "current_subtask_index": normalized_index,
        "completed_subtasks": completed_subtasks,
        "next_subtask": next_subtask,
        "confidence": confidence,
        "evidence": evidence,
        "memory_update": memory_update,
        "state_summary": state_summary,
    }


def _enforce_task_sequence(
    decision: dict[str, Any],
    memory: RollingTaskMemory,
    task_spec: OrderedTaskSpec,
) -> dict[str, Any]:
    corrected = dict(decision)
    memory_completed_count = memory.highest_completed_count(task_spec)

    if corrected["current_subtask"] == task_spec.done_label:
        corrected["current_subtask_index"] = task_spec.done_index
        corrected["completed_subtasks"] = list(task_spec.subtasks)
        corrected["next_subtask"] = task_spec.done_label
        corrected["sequence_enforced"] = False
        return corrected

    current_index = int(corrected["current_subtask_index"])
    predicted_completed_len = len(
        task_spec.normalize_completed_prefix(corrected["completed_subtasks"])
    )
    allowed_completed_len = min(predicted_completed_len, memory_completed_count + 1)
    minimum_index = max(memory_completed_count, allowed_completed_len)

    enforced = False
    if minimum_index > current_index:
        enforced = True
        current_index = minimum_index
    elif current_index > memory_completed_count + 1:
        enforced = True
        current_index = min(current_index, memory_completed_count + 1)

    if current_index >= task_spec.done_index:
        corrected["current_subtask"] = task_spec.done_label
        corrected["current_subtask_index"] = task_spec.done_index
        corrected["completed_subtasks"] = list(task_spec.subtasks)
        corrected["next_subtask"] = task_spec.done_label
    else:
        corrected["current_subtask"] = task_spec.label_from_index(current_index)
        corrected["current_subtask_index"] = current_index
        corrected["completed_subtasks"] = task_spec.prefix(current_index)
        corrected["next_subtask"] = task_spec.next_after_current(current_index)

    corrected["sequence_enforced"] = enforced
    return corrected


def build_system_prompt(task_spec: OrderedTaskSpec) -> str:
    return (
        "You are a long-horizon task-stage estimator and working memory module "
        "for a mobile manipulation robot.\n"
        f"The overall task is fixed: {task_spec.total_task}\n"
        "Interpret the scene strictly using the ordered subtask list below:\n"
        f"{task_spec.ordered_subtasks_text()}\n"
        "Based on the current images, robot state, and rolling memory, decide which "
        "subtask is active right now.\n"
        "Important instructions:\n"
        "1. In `state_summary`, describe key object locations and environment state in detail. "
        "This is critical for reasoning about later subtasks such as returning objects to their original place.\n"
        "2. The robot grippers are blue. Do not mistake the grippers themselves for bread, lettuce, "
        "sponges, or other task objects.\n"
        "3. Unless there is strong evidence, do not regress to earlier subtasks. Follow the ordered list "
        "strictly and do not skip multiple unfinished subtasks.\n"
        "4. `completed_subtasks` must be a prefix of the ordered list.\n"
        f"5. `current_subtask` must be one of the listed subtasks, `{task_spec.done_label}`, or `{task_spec.uncertain_label}`.\n"
        "6. `memory_update` must contain only useful new facts or state changes; do not restate the prompt.\n"
        "Output exactly one JSON object. No markdown. No extra explanation.\n"
        "JSON schema:\n"
        "{\n"
        '  "current_subtask": "ordered subtask | task complete | uncertain",\n'
        '  "current_subtask_index": "0-based subtask index; use subtask count when complete; -1 when uncertain",\n'
        '  "completed_subtasks": ["ordered completed prefix"],\n'
        '  "next_subtask": "next likely subtask | task complete | uncertain",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "concrete image/state evidence",\n'
        '  "memory_update": "new fact worth keeping in rolling memory",\n'
        '  "state_summary": "scene summary including object locations"\n'
        "}\n"
        "confidence must be a decimal between 0 and 1."
    )


def build_user_prompt(
    task_spec: OrderedTaskSpec,
    *,
    step_index: int,
    total_steps: int,
    camera_names: tuple[str, ...],
    observation_text: str,
    memory_text: str,
) -> str:
    return (
        f"Overall task: {task_spec.total_task}\n"
        f"Current replay step: {step_index} / {max(total_steps - 1, 0)}\n"
        f"Camera views: {', '.join(camera_names)}\n"
        "Ordered subtasks:\n"
        f"{task_spec.ordered_subtasks_text()}\n\n"
        f"Current observation:\n{observation_text}\n\n"
        f"Rolling memory:\n{memory_text}\n\n"
        "Decide which subtask is active now and store any durable new fact in memory_update.\n"
        "Output JSON only."
    )


class ReplayOrderedTaskMemoryRuntime:
    """Ordered task-stage analysis runtime for replay hybrid planning.

    Works with any environment that exposes ``get_image(cam, idx)``, ``get_cursor()``,
    ``num_steps``, and ``camera_names`` — including ``PiperRealEnvironment`` on the
    real robot (duck-typed, not limited to offline replay).
    """

    def __init__(
        self,
        replay_environment: ReplayEnvironment,
        config: PlannerConfig,
    ) -> None:
        from openai import OpenAI

        task_spec_path = config.task_spec_path.strip()
        if not task_spec_path:
            raise ValueError("planner.task_spec_path must be set to enable ordered task memory")

        self.replay_environment = replay_environment
        self.config = config
        self.task_spec = load_ordered_task_spec(task_spec_path)
        self.memory = RollingTaskMemory(max_entries=config.task_memory_max_entries)
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._last_step_index: int | None = None
        self._last_decision: TaskStageDecision | None = None

    def build_context(self) -> dict[str, str]:
        decision = self.observe()
        return {
            "ordered_task_spec_text": self.task_spec.as_prompt_text(),
            "working_memory_text": self.memory.as_prompt_text(self.task_spec),
            "stage_estimate_text": json.dumps(decision.as_json_dict(), ensure_ascii=False),
        }

    def observe(self, *, force_refresh: bool = False) -> TaskStageDecision:
        if self.replay_environment.num_steps <= 0:
            raise TaskStageAnalysisError("replay dataset is empty")

        step_index = self._current_frame_index()
        if (
            not force_refresh
            and self._last_step_index == step_index
            and self._last_decision is not None
        ):
            return self._last_decision

        decision = self._query_stage(step_index)
        self.memory.add(
            TaskMemoryEntry(
                step_index=step_index,
                current_subtask=decision.current_subtask,
                current_subtask_index=decision.current_subtask_index,
                completed_subtasks=list(decision.completed_subtasks),
                next_subtask=decision.next_subtask,
                confidence=decision.confidence,
                evidence=decision.evidence,
                memory_update=decision.memory_update,
                state_summary=decision.state_summary,
            )
        )
        self._last_step_index = step_index
        self._last_decision = decision
        return decision

    def _query_stage(self, step_index: int) -> TaskStageDecision:
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_user_prompt(
                    self.task_spec,
                    step_index=step_index,
                    total_steps=self.replay_environment.num_steps,
                    camera_names=self.replay_environment.camera_names,
                    observation_text=self._build_observation_text(step_index),
                    memory_text=self.memory.as_prompt_text(self.task_spec),
                ),
            }
        ]
        for cam_name in self.replay_environment.camera_names:
            message_content.append({"type": "text", "text": f"Camera view: {cam_name}"})
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(cam_name, step_index)},
                }
            )

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            max_tokens=self.config.task_memory_max_tokens,
            messages=[
                {"role": "system", "content": build_system_prompt(self.task_spec)},
                {"role": "user", "content": message_content},
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.config.task_memory_enable_thinking,
                }
            },
        )

        try:
            raw_text, raw_json = extract_message_json_text(response.choices[0].message)
        except ValueError as exc:
            raise TaskStageAnalysisError(str(exc)) from exc

        parsed: dict[str, Any] = _extract_decision_from_text(raw_text, self.task_spec)
        try:
            parsed = {**parsed, **_extract_json_object(raw_json)}
        except ValueError:
            pass

        try:
            decision_data = _normalize_decision(parsed, raw_text, self.task_spec, self.memory)
            decision_data = _enforce_task_sequence(decision_data, self.memory, self.task_spec)
        except Exception as exc:  # noqa: BLE001
            raise TaskStageAnalysisError(f"failed to normalize ordered task decision: {exc}") from exc

        decision = TaskStageDecision(
            current_subtask=str(decision_data["current_subtask"]),
            current_subtask_index=int(decision_data["current_subtask_index"]),
            completed_subtasks=list(decision_data["completed_subtasks"]),
            next_subtask=str(decision_data["next_subtask"]),
            confidence=float(decision_data["confidence"]),
            evidence=str(decision_data["evidence"]),
            memory_update=str(decision_data["memory_update"]),
            state_summary=str(decision_data["state_summary"]),
            sequence_enforced=bool(decision_data.get("sequence_enforced", False)),
        )
        logging.info(
            "Replay ordered task stage estimate at step %d: %s",
            step_index,
            json.dumps(decision.as_json_dict(), ensure_ascii=False),
        )
        return decision

    def _build_observation_text(self, step_index: int) -> str:
        state = None
        if hasattr(self.replay_environment, "get_state"):
            state = self.replay_environment.get_state(step_index)
        action = None
        if hasattr(self.replay_environment, "get_ground_truth_action"):
            action = self.replay_environment.get_ground_truth_action(step_index)

        lines: list[str] = []
        rounded_state = _round_values(state)
        if rounded_state is not None:
            lines.append(f"state={rounded_state}")
        rounded_action = _round_values(action)
        if rounded_action is not None:
            lines.append(f"ground_truth_action={rounded_action}")
        if not lines:
            lines.append("No structured state is available for this replay step.")
        return "\n".join(lines)

    def _current_frame_index(self) -> int:
        cursor = self.replay_environment.get_cursor()
        if cursor >= self.replay_environment.num_steps:
            return self.replay_environment.num_steps - 1
        return cursor

    def _encode_image(self, cam_name: str, step_index: int) -> str:
        frame = self.replay_environment.get_image(cam_name, step_index)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise TaskStageAnalysisError(
                f"failed to encode replay frame {step_index} from {cam_name}"
            )
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"
