"""Shared LLM response parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_DECODER = json.JSONDecoder()


def extract_json_text(raw_text: str) -> str:
    """Extract a JSON object from LLM text that may contain wrappers."""
    stripped = raw_text.strip()
    stripped = _THINK_RE.sub("", stripped).strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence_wrapper(stripped)

    for start in _iter_json_object_starts(stripped):
        try:
            payload, end = _JSON_DECODER.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return stripped[start : start + end]

    raise ValueError("LLM response did not contain a valid JSON object")


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


def _strip_code_fence_wrapper(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _iter_json_object_starts(text: str):
    for idx, char in enumerate(text):
        if char == "{":
            yield idx
