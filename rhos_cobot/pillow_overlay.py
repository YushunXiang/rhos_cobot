"""Shared Pillow-based overlay rendering for feat-gui video producers.

This module provides font discovery, image-array conversion, and
drawing primitives built on Pillow's ``ImageDraw``, replacing the
OpenCV overlay paths in the feat-gui video producers. See
``docs/superpowers/specs/2026-04-14-pillow-overlay-migration-design.md``
for the full design.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont


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


@functools.lru_cache(maxsize=32)
def load_font(size: int, font_path: Path) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at the given pixel size."""
    return ImageFont.truetype(str(font_path), size)


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a BGR uint8 ndarray into an RGB PIL image."""
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got dtype={frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) frame, got shape={frame.shape}")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError(f"empty frame: shape={frame.shape}")
    return Image.fromarray(frame[..., ::-1].copy(), mode="RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into a BGR uint8 ndarray."""
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    return np.asarray(rgb)[..., ::-1].copy()
