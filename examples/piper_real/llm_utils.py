"""Shared LLM response parsing utilities."""

from __future__ import annotations

import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json_text(raw_text: str) -> str:
    """Extract a JSON object from LLM text that may contain wrappers."""
    stripped = raw_text.strip()
    stripped = _THINK_RE.sub("", stripped).strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return stripped[start : end + 1]


def extract_message_json_text(message: Any) -> tuple[str, str]:
    """Select the first assistant message field that contains a JSON object."""
    for candidate in _iter_message_text_candidates(message):
        try:
            return candidate, extract_json_text(candidate)
        except ValueError:
            continue
    raise ValueError("LLM response did not contain a JSON object in content or reasoning fields")


def _iter_message_text_candidates(message: Any):
    seen: set[str] = set()
    for field_name in ("content", "reasoning_content", "reasoning"):
        value = message.get(field_name) if isinstance(message, dict) else getattr(message, field_name, None)
        for text in _iter_text_fragments(value):
            if text not in seen:
                seen.add(text)
                yield text


def _iter_text_fragments(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_text_fragments(item)
        return
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning"):
            if key in value:
                yield from _iter_text_fragments(value[key])
        return

    for attr in ("text", "content", "reasoning"):
        nested = getattr(value, attr, None)
        if nested is not None:
            yield from _iter_text_fragments(nested)
