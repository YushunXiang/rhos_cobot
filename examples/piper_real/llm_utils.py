"""Shared LLM response parsing utilities."""


def extract_json_text(raw_text: str) -> str:
    """Extract JSON object from an LLM response that may contain markdown fences."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return stripped[start : end + 1]
