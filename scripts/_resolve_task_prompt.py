#!/usr/bin/env python3
"""Resolve a task-level prompt from the local task prompt catalog."""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: scripts/_resolve_task_prompt.py <catalog_path> <task_name>",
            file=sys.stderr,
        )
        return 2

    catalog_path = pathlib.Path(sys.argv[1])
    task_name = sys.argv[2]

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Task prompt catalog does not exist: {catalog_path}", file=sys.stderr)
        return 1

    task_entry = catalog.get(task_name)
    if not isinstance(task_entry, dict):
        available = ", ".join(sorted(catalog))
        print(
            f"Unknown TASK_NAME '{task_name}'. Available tasks: {available}",
            file=sys.stderr,
        )
        return 1

    prompt = task_entry.get("prompt", "").strip()
    if not prompt:
        print(f"Task '{task_name}' does not define a top-level prompt.", file=sys.stderr)
        return 1

    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
