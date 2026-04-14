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
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def max_text_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    texts: Iterable[str],
    *,
    padding_x: int,
) -> int:
    """Return max(text_width + 2*padding_x) across non-empty texts."""
    widest = 0
    for text in texts:
        if not text:
            continue
        bbox = draw.textbbox((0, 0), text, font=font)
        widest = max(widest, (bbox[2] - bbox[0]) + 2 * padding_x)
    return widest


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    padding: tuple[int, int] = (8, 5),
    fg: tuple[int, ...] = (245, 245, 245),
    bg: tuple[int, ...] | None = None,
    radius: int = 0,
    box_width: int | None = None,
) -> tuple[int, int, int, int]:
    """Draw text and an optional background box, returning its bbox."""
    x, y = xy
    pad_x, pad_y = padding
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    rect_width = box_width if box_width is not None else (text_w + 2 * pad_x)
    x0 = x - pad_x
    y0 = text_bbox[1] - pad_y
    x1 = x0 + rect_width
    y1 = text_bbox[3] + pad_y
    if bg is not None:
        if radius > 0:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=bg)
        else:
            draw.rectangle((x0, y0, x1, y1), fill=bg)
    draw.text((x, y), text, font=font, fill=fg)
    return (x0, y0, x1, y1)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    color: tuple[int, ...],
    width: int = 2,
) -> None:
    """Draw a polyline through points."""
    if len(points) < 2:
        return
    draw.line(list(points), fill=color, width=width, joint="curve")


def draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    color: tuple[int, ...],
    radius: int,
    outline: tuple[int, ...] | None = None,
    outline_width: int = 0,
) -> None:
    """Draw a filled circle with an optional outline."""
    x, y = xy
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=color)
    if outline is not None and outline_width > 0:
        draw.ellipse(bbox, outline=outline, width=outline_width)
