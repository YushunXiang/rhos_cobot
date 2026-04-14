"""Shared Pillow-based overlay rendering for feat-gui video producers.

This module provides font discovery, image-array conversion, and
drawing primitives built on Pillow's ``ImageDraw``, replacing the
OpenCV overlay paths in the feat-gui video producers. See
``docs/superpowers/specs/2026-04-14-pillow-overlay-migration-design.md``
for the full design.
"""
from __future__ import annotations

import os
from pathlib import Path


_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
)

_LATIN_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

_FONT_ENV_VAR: str = "RHOS_COBOT_VIDEO_FONT"


class FontUnavailableError(RuntimeError):
    """Raised when no usable TrueType font can be resolved."""


def resolve_font_path(user_path: Path | None = None) -> Path:
    """Resolve a TrueType font path."""
    if user_path is not None:
        candidate = Path(user_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Font file not found: {candidate}")
        return candidate

    env_value = os.environ.get(_FONT_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"${_FONT_ENV_VAR} points to missing file: {candidate}")
        return candidate

    for raw in _CJK_FONT_CANDIDATES:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate

    for raw in _LATIN_FONT_CANDIDATES:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate

    raise FontUnavailableError(
        "No usable font found. To fix: "
        "(1) install a CJK font (e.g., `apt install fonts-noto-cjk`), "
        f"(2) set ${_FONT_ENV_VAR}=/path/to/font.ttf, "
        "(3) pass --video-font-path on the CLI."
    )
