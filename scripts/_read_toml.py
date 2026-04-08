#!/usr/bin/env python3
"""Read a value from a TOML file by dotted key path.

Usage:
    python3 scripts/_read_toml.py <toml_file> <key> [<key> ...]

Examples:
    python3 scripts/_read_toml.py config/servers.toml pi0.port
    python3 scripts/_read_toml.py config/servers.toml vllm.remote.host vllm.remote.session_name

Supports Python 3.10+ (uses tomllib on 3.11+, otherwise a built-in parser
for the simple TOML subset used by config/servers.toml).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


def _example_path(path: str) -> Path:
    toml_path = Path(path)
    return toml_path.with_name(f"{toml_path.stem}.example{toml_path.suffix}")


def _ensure_toml_exists(path: str) -> None:
    toml_path = Path(path)
    if toml_path.exists():
        return

    example_path = _example_path(path)
    print(f"error: TOML file '{toml_path}' not found.", file=sys.stderr)
    if example_path.exists():
        print(
            "Create it from the example template and update the values for your environment:",
            file=sys.stderr,
        )
        print(f"  cp {example_path} {toml_path}", file=sys.stderr)
    sys.exit(1)


def _load_toml(path: str) -> dict:
    """Try tomllib/tomli first; fall back to a minimal parser."""
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass

    # Minimal parser: handles [section.subsection] headers and key = "value" / key = number lines.
    data: dict = {}
    current: dict = data
    section_path: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header: [a.b.c]
            m = re.match(r"^\[([^\]]+)\]$", line)
            if m:
                section_path = m.group(1).split(".")
                current = data
                for part in section_path:
                    current = current.setdefault(part, {})
                continue

            # Key = value
            m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if m:
                key = m.group(1)
                raw_value = m.group(2).strip()
                # Quoted string
                if (raw_value.startswith('"') and raw_value.endswith('"')) or (
                    raw_value.startswith("'") and raw_value.endswith("'")
                ):
                    current[key] = raw_value[1:-1]
                # Integer
                elif re.match(r'^-?\d+$', raw_value):
                    current[key] = int(raw_value)
                # Float
                elif re.match(r'^-?\d+\.\d+$', raw_value):
                    current[key] = float(raw_value)
                # Boolean
                elif raw_value in ("true", "false"):
                    current[key] = raw_value == "true"
                else:
                    current[key] = raw_value

    return data


def resolve(data: dict, key: str) -> str:
    parts = key.split(".")
    obj = data
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            print(f"error: key '{key}' not found", file=sys.stderr)
            sys.exit(1)
        obj = obj[part]
    return str(obj)


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <toml_file> <key> [<key> ...]", file=sys.stderr)
        sys.exit(2)

    toml_path = sys.argv[1]
    keys = sys.argv[2:]

    _ensure_toml_exists(toml_path)
    data = _load_toml(toml_path)

    for key in keys:
        print(resolve(data, key))


if __name__ == "__main__":
    main()
