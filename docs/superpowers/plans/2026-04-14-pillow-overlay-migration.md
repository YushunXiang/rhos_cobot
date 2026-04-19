# Pillow Overlay Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `cv2.putText`/`cv2.line`/`cv2.circle`/`cv2.rectangle`/`cv2.addWeighted`-based overlay drawing in three feat-gui video producers (`replay_visualizer.py`, `visualize_replay_progress.py`, `extract_replay_camera_video.py`) with a shared Pillow helper module, gaining higher-quality text rendering and native CJK support.

**Architecture:** New shared module `rhos_cobot/pillow_overlay.py` holds font discovery (CJK-preferred, Latin fallback, `$RHOS_COBOT_VIDEO_FONT` env var) and Pillow-based drawing primitives. Three consumer files migrate their overlay paths to use it. `cv2.VideoWriter` and solid-fill canvas construction stay unchanged.

**Tech Stack:** Pillow 11.2.1 (already declared in `examples/piper_real/requirements.in`/`.txt`), numpy, pytest. No new runtime dependencies.

**Reference spec:** `docs/superpowers/specs/2026-04-14-pillow-overlay-migration-design.md`

---

## Phase A — Build `rhos_cobot/pillow_overlay.py` (TDD)

### Task 1: Module skeleton + `FontUnavailableError`

**Files:**
- Create: `rhos_cobot/pillow_overlay.py`
- Create: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pillow_overlay.py`:

```python
"""Tests for rhos_cobot.pillow_overlay."""
from __future__ import annotations

import pytest


def test_module_exports_font_unavailable_error():
    from rhos_cobot import pillow_overlay as po

    assert issubclass(po.FontUnavailableError, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pillow_overlay.py::test_module_exports_font_unavailable_error -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rhos_cobot.pillow_overlay'`

- [ ] **Step 3: Write minimal implementation**

`rhos_cobot/pillow_overlay.py`:

```python
"""Shared Pillow-based overlay rendering for feat-gui video producers.

This module provides font discovery, image-array conversion, and
drawing primitives built on Pillow's `ImageDraw`, replacing the
`cv2.putText`/`cv2.line`/`cv2.circle`-based overlay paths in the
three feat-gui video producers. See
`docs/superpowers/specs/2026-04-14-pillow-overlay-migration-design.md`
for the full design.
"""
from __future__ import annotations


class FontUnavailableError(RuntimeError):
    """Raised when no usable TrueType font can be resolved."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pillow_overlay.py::test_module_exports_font_unavailable_error -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): scaffold rhos_cobot.pillow_overlay module"
```

---

### Task 2: `resolve_font_path` with user → env → CJK → Latin priority

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pillow_overlay.py`:

```python
from pathlib import Path


def test_resolve_font_user_path_exists(tmp_path: Path):
    from rhos_cobot import pillow_overlay as po

    font_file = tmp_path / "custom.ttf"
    font_file.write_bytes(b"")
    assert po.resolve_font_path(font_file) == font_file


def test_resolve_font_user_path_missing_raises(tmp_path: Path):
    from rhos_cobot import pillow_overlay as po

    missing = tmp_path / "nope.ttf"
    with pytest.raises(FileNotFoundError, match="Font file not found"):
        po.resolve_font_path(missing)


def test_resolve_font_env_var_used(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    font_file = tmp_path / "env.ttf"
    font_file.write_bytes(b"")
    monkeypatch.setenv(po._FONT_ENV_VAR, str(font_file))
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", ())
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", ())
    assert po.resolve_font_path(None) == font_file


def test_resolve_font_env_var_missing_file_raises(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    monkeypatch.setenv(po._FONT_ENV_VAR, str(tmp_path / "nope.ttf"))
    with pytest.raises(FileNotFoundError, match=po._FONT_ENV_VAR):
        po.resolve_font_path(None)


def test_resolve_font_cjk_preferred_over_latin(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    cjk = tmp_path / "cjk.ttc"
    latin = tmp_path / "latin.ttf"
    cjk.write_bytes(b"")
    latin.write_bytes(b"")
    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(cjk),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(latin),))
    assert po.resolve_font_path(None) == cjk


def test_resolve_font_falls_back_to_latin(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    latin = tmp_path / "latin.ttf"
    latin.write_bytes(b"")
    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(tmp_path / "missing-cjk.ttc"),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(latin),))
    assert po.resolve_font_path(None) == latin


def test_resolve_font_all_missing_raises(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(tmp_path / "a.ttc"),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(tmp_path / "b.ttf"),))
    with pytest.raises(po.FontUnavailableError, match="No usable font found"):
        po.resolve_font_path(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_overlay.py -v -k "resolve_font"`
Expected: 7 FAILED (AttributeError / missing `resolve_font_path`, `_FONT_ENV_VAR`, `_CJK_FONT_CANDIDATES`, `_LATIN_FONT_CANDIDATES`)

- [ ] **Step 3: Implement**

Append to `rhos_cobot/pillow_overlay.py`:

```python
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


def resolve_font_path(user_path: Path | None = None) -> Path:
    """Resolve a TrueType font path.

    Priority (first existing wins):
        1. ``user_path`` argument
        2. ``$RHOS_COBOT_VIDEO_FONT`` environment variable
        3. CJK candidates
        4. Latin candidates
        5. raise :class:`FontUnavailableError`
    """
    if user_path is not None:
        candidate = Path(user_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Font file not found: {candidate}")
        return candidate

    env_value = os.environ.get(_FONT_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"${_FONT_ENV_VAR} points to missing file: {candidate}"
            )
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v -k "resolve_font"`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add resolve_font_path with CJK/Latin/env fallbacks"
```

---

### Task 3: `load_font` with LRU cache

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pillow_overlay.py`:

```python
def test_load_font_lru_cached(tmp_path: Path, monkeypatch):
    from rhos_cobot import pillow_overlay as po

    class FakeFont:
        pass

    calls: list[tuple[str, int]] = []

    def fake_truetype(path, size):
        calls.append((str(path), size))
        return FakeFont()

    monkeypatch.setattr(po.ImageFont, "truetype", fake_truetype)
    po.load_font.cache_clear()

    font_path = tmp_path / "fake.ttf"
    a = po.load_font(14, font_path)
    b = po.load_font(14, font_path)
    c = po.load_font(18, font_path)

    assert a is b
    assert a is not c
    assert len(calls) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pillow_overlay.py::test_load_font_lru_cached -v`
Expected: FAIL (`AttributeError: module 'rhos_cobot.pillow_overlay' has no attribute 'ImageFont'` or `load_font`)

- [ ] **Step 3: Implement**

Add at the top-of-module imports in `rhos_cobot/pillow_overlay.py` (replace the `from __future__` line group):

```python
from __future__ import annotations

import functools
import os
from pathlib import Path

from PIL import ImageFont
```

Append to the module body (after `resolve_font_path`):

```python
@functools.lru_cache(maxsize=32)
def load_font(size: int, font_path: Path) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at the given pixel size (LRU-cached by (size, path))."""
    return ImageFont.truetype(str(font_path), size)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v`
Expected: all prior tests + new one PASS

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add lru-cached load_font"
```

---

### Task 4: `bgr_to_pil` / `pil_to_bgr` conversion

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pillow_overlay.py`:

```python
import numpy as np


def test_bgr_to_pil_color_swap():
    from rhos_cobot import pillow_overlay as po

    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)  # BGR red
    pil = po.bgr_to_pil(bgr)
    assert pil.size == (1, 1)
    assert pil.mode == "RGB"
    assert pil.getpixel((0, 0)) == (255, 0, 0)


def test_pil_to_bgr_roundtrip():
    from rhos_cobot import pillow_overlay as po

    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    restored = po.pil_to_bgr(po.bgr_to_pil(bgr))
    assert np.array_equal(restored, bgr)


def test_bgr_to_pil_invalid_shape_raises():
    from rhos_cobot import pillow_overlay as po

    with pytest.raises(ValueError, match=r"expected \(H, W, 3\)"):
        po.bgr_to_pil(np.zeros((3, 3), dtype=np.uint8))


def test_bgr_to_pil_invalid_dtype_raises():
    from rhos_cobot import pillow_overlay as po

    with pytest.raises(ValueError, match="dtype"):
        po.bgr_to_pil(np.zeros((2, 2, 3), dtype=np.float32))


def test_bgr_to_pil_empty_shape_raises():
    from rhos_cobot import pillow_overlay as po

    with pytest.raises(ValueError, match="empty"):
        po.bgr_to_pil(np.zeros((0, 2, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_overlay.py -v -k "bgr_to_pil or pil_to_bgr"`
Expected: 5 FAILED (missing attributes)

- [ ] **Step 3: Implement**

Update the import block in `rhos_cobot/pillow_overlay.py` to add `Image`:

```python
from PIL import Image, ImageFont
```

Also add `import numpy as np` near the top imports. Then append to the module body:

```python
def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a BGR uint8 ndarray into an RGB PIL Image (copy)."""
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got dtype={frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) frame, got shape={frame.shape}")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError(f"empty frame: shape={frame.shape}")
    return Image.fromarray(frame[..., ::-1].copy(), mode="RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert an RGB(A) PIL Image into a BGR uint8 ndarray (copy)."""
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    return np.asarray(rgb)[..., ::-1].copy()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add bgr_to_pil / pil_to_bgr conversion helpers"
```

---

### Task 5: `draw_text_box` + `max_text_width`

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pillow_overlay.py`:

```python
from PIL import Image, ImageDraw, ImageFont


def _default_font():
    return ImageFont.load_default()


def test_draw_text_box_writes_pixels():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (200, 80), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_text_box(draw, (10, 30), "HELLO", _default_font(), fg=(255, 255, 255))
    assert np.asarray(img).sum() > 0


def test_draw_text_box_returns_bbox_within_image():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (200, 80), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = po.draw_text_box(draw, (10, 20), "X", _default_font())
    x0, y0, x1, y1 = bbox
    assert x0 < x1
    assert y0 < y1
    assert 0 <= x0 and x1 <= 200
    assert y1 <= 80


def test_draw_text_box_uniform_box_width():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (300, 80), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = po.draw_text_box(
        draw, (10, 20), "x", _default_font(),
        bg=(50, 50, 50), box_width=200,
    )
    assert bbox[2] - bbox[0] == 200


def test_draw_text_box_with_bg_paints_background():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (100, 40), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_text_box(
        draw, (10, 15), "Y", _default_font(),
        fg=(255, 255, 255), bg=(40, 80, 160),
    )
    arr = np.asarray(img)
    # at least one pixel should match the bg color (blue channel dominant)
    blue_pixels = np.all(arr == (40, 80, 160), axis=-1)
    assert blue_pixels.any()


def test_max_text_width_returns_widest():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (300, 80))
    draw = ImageDraw.Draw(img)
    wide = po.max_text_width(draw, _default_font(),
                              ["a", "longer string", "b"], padding_x=4)
    short = po.max_text_width(draw, _default_font(), ["a"], padding_x=4)
    assert wide > short


def test_max_text_width_ignores_empty_strings():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (100, 40))
    draw = ImageDraw.Draw(img)
    assert po.max_text_width(draw, _default_font(), ["", ""], padding_x=4) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_overlay.py -v -k "text_box or text_width"`
Expected: 6 FAILED (missing `draw_text_box`, `max_text_width`)

- [ ] **Step 3: Implement**

Update imports in `rhos_cobot/pillow_overlay.py`:

```python
from PIL import Image, ImageDraw, ImageFont
```

Add `from typing import Iterable, Sequence` to imports.

Append to module body:

```python
def max_text_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    texts: Iterable[str],
    *,
    padding_x: int,
) -> int:
    """Return max(text_width + 2*padding_x) across non-empty texts, or 0."""
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
    """Draw text at ``xy`` (text top-left). If ``bg`` is set, paint a filled
    rectangle behind the text. ``bg`` and ``fg`` accept 3-tuple (RGB) or
    4-tuple (RGBA) — use 4-tuple when drawing onto an RGBA overlay for
    translucent backgrounds. ``radius`` > 0 produces a rounded rectangle.
    If ``box_width`` is provided, the background rectangle uses that exact
    width; otherwise it sizes to text + padding.

    Returns the full bounding box ``(x0, y0, x1, y1)`` of the background
    rectangle so callers can stack lines.
    """
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add draw_text_box + max_text_width"
```

---

### Task 6: `draw_polyline` + `draw_marker`

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pillow_overlay.py`:

```python
def test_draw_polyline_renders():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (40, 40), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_polyline(draw, [(5, 5), (35, 35)], color=(0, 255, 0), width=2)
    arr = np.asarray(img)
    # some green pixels along the diagonal
    assert arr[:, :, 1].sum() > 0
    # red and blue channels remain zero
    assert arr[:, :, 0].sum() == 0
    assert arr[:, :, 2].sum() == 0


def test_draw_polyline_single_point_is_noop():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (10, 10), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_polyline(draw, [(5, 5)], color=(255, 0, 0), width=2)
    assert np.asarray(img).sum() == 0


def test_draw_polyline_empty_is_noop():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (10, 10), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_polyline(draw, [], color=(255, 0, 0), width=2)
    assert np.asarray(img).sum() == 0


def test_draw_marker_fills_center():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (20, 20), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_marker(draw, (10, 10), color=(255, 0, 0), radius=4)
    arr = np.asarray(img)
    assert arr[10, 10, 0] == 255
    assert arr[0, 0].sum() == 0


def test_draw_marker_with_outline_adds_outline_pixels():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (30, 30), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_marker(
        draw, (15, 15),
        color=(255, 0, 0), radius=5,
        outline=(0, 0, 255), outline_width=2,
    )
    arr = np.asarray(img)
    # blue pixels exist somewhere (the outline ring)
    assert arr[:, :, 2].sum() > 0
    # red center still set
    assert arr[15, 15, 0] == 255
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_overlay.py -v -k "polyline or marker"`
Expected: 5 FAILED (missing `draw_polyline`, `draw_marker`)

- [ ] **Step 3: Implement**

Append to `rhos_cobot/pillow_overlay.py`:

```python
def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    color: tuple[int, ...],
    width: int = 2,
) -> None:
    """Draw a polyline through ``points`` (at least 2). Rounded joins when
    ``width >= 2``. Note: Pillow's line rasterization is aliased; if
    polyline smoothness becomes an issue, wrap this in a 2x supersample
    pass (not implemented in V1 — see spec §9.1).
    """
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
    """Draw a filled circle (anti-aliased by Pillow) with optional outline ring."""
    x, y = xy
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=color)
    if outline is not None and outline_width > 0:
        draw.ellipse(bbox, outline=outline, width=outline_width)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add draw_polyline + draw_marker"
```

---

### Task 7: `new_overlay` + `composite_overlay_on_bgr`

**Files:**
- Modify: `rhos_cobot/pillow_overlay.py`
- Modify: `tests/test_pillow_overlay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pillow_overlay.py`:

```python
def test_new_overlay_is_transparent_rgba():
    from rhos_cobot import pillow_overlay as po

    overlay, draw = po.new_overlay((30, 20))
    assert overlay.mode == "RGBA"
    assert overlay.size == (30, 20)
    assert overlay.getpixel((15, 10)) == (0, 0, 0, 0)
    # draw is bound to overlay
    draw.rectangle((0, 0, 30, 20), fill=(255, 255, 255, 255))
    assert overlay.getpixel((15, 10)) == (255, 255, 255, 255)


def test_composite_overlay_leaves_uncovered_pixels_intact():
    from rhos_cobot import pillow_overlay as po

    # BGR green frame, overlay paints red only in center
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[..., 1] = 255  # G channel (BGR green)
    overlay, draw = po.new_overlay((20, 20))
    draw.rectangle((5, 5, 14, 14), fill=(255, 0, 0, 255))

    result = po.composite_overlay_on_bgr(frame, overlay)
    assert tuple(result[0, 0]) == (0, 255, 0)   # corner still pure BGR green
    assert tuple(result[10, 10]) == (0, 0, 255)  # center is BGR red


def test_composite_overlay_blends_alpha():
    from rhos_cobot import pillow_overlay as po

    # BGR green frame + 50%-alpha red overlay in center
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[..., 1] = 255
    overlay, draw = po.new_overlay((20, 20))
    draw.rectangle((5, 5, 14, 14), fill=(255, 0, 0, 128))

    result = po.composite_overlay_on_bgr(frame, overlay)
    c = result[10, 10]
    assert c[0] == 0          # B
    assert 100 < c[1] < 200   # G reduced
    assert 100 < c[2] < 200   # R added


def test_composite_overlay_does_not_mutate_input_frame():
    from rhos_cobot import pillow_overlay as po

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[..., 1] = 200
    frame_before = frame.copy()
    overlay, draw = po.new_overlay((10, 10))
    draw.rectangle((0, 0, 10, 10), fill=(255, 0, 0, 128))
    po.composite_overlay_on_bgr(frame, overlay)
    assert np.array_equal(frame, frame_before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_overlay.py -v -k "overlay or composite"`
Expected: 4 FAILED (missing `new_overlay`, `composite_overlay_on_bgr`)

- [ ] **Step 3: Implement**

Append to `rhos_cobot/pillow_overlay.py`:

```python
def new_overlay(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Return ``(overlay, draw)`` where overlay is a fully-transparent RGBA image."""
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    return overlay, ImageDraw.Draw(overlay)


def composite_overlay_on_bgr(
    frame: np.ndarray,
    overlay: Image.Image,
) -> np.ndarray:
    """Alpha-composite an RGBA overlay onto a BGR uint8 ndarray frame.
    Returns a new BGR ndarray (does not mutate input)."""
    base = bgr_to_pil(frame).convert("RGBA")
    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return pil_to_bgr(composited)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pillow_overlay.py -v`
Expected: all tests PASS (full suite)

- [ ] **Step 5: Commit**

```bash
git add rhos_cobot/pillow_overlay.py tests/test_pillow_overlay.py
git commit -m "feat(overlay): add new_overlay + composite_overlay_on_bgr"
```

---

## Phase B — Migrate consumer files

### Task 8: Migrate `scripts/extract_replay_camera_video.py`

Smallest consumer. Extract tile labeling into a pure helper for testability, then swap it to Pillow and add the `--video-font-path` CLI flag.

**Files:**
- Modify: `scripts/extract_replay_camera_video.py`
- Modify: `tests/test_extract_replay_camera_video.py`

- [ ] **Step 1: Read existing code to locate the tile-labeling block**

Run: `grep -n "cv2.putText\|cv2.rectangle" scripts/extract_replay_camera_video.py`
Expected lines around 141–152 showing the `cv2.rectangle` black bar + `cv2.putText` label on each tile.

- [ ] **Step 2: Write failing test for the new helper**

Append to `tests/test_extract_replay_camera_video.py`:

```python
def test_annotate_tile_writes_label_with_black_background(monkeypatch):
    """The new helper must paint a black bg strip and white label text."""
    from pathlib import Path

    module = _load_module(monkeypatch)

    from PIL import ImageFont  # real PIL, not stubbed
    font = ImageFont.load_default()

    tile = np.full((80, 160, 3), 200, dtype=np.uint8)  # BGR grey
    out = module._annotate_tile(tile, "frame 5  t=0.2s", font, tile_width=160)

    assert out.shape == tile.shape
    assert out.dtype == np.uint8
    # Top strip has black pixels (bg) — must exist somewhere in the top band
    top_band = out[:28]
    assert (top_band.sum(axis=-1) == 0).any()
    # White text pixels must exist somewhere in the top band
    assert (top_band.sum(axis=-1) >= 3 * 250).any()
    # The bottom of the tile should be untouched (still grey)
    np.testing.assert_array_equal(out[40:], tile[40:])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_extract_replay_camera_video.py::test_annotate_tile_writes_label_with_black_background -v`
Expected: FAIL (`AttributeError: module ... has no attribute '_annotate_tile'`)

- [ ] **Step 4: Implement — add helper + CLI flag + swap call site**

Edit `scripts/extract_replay_camera_video.py`:

4a. Update imports (after the existing imports):

```python
from rhos_cobot.pillow_overlay import (
    bgr_to_pil,
    draw_text_box,
    load_font,
    pil_to_bgr,
    resolve_font_path,
)
from PIL import ImageFont
```

4b. Add the CLI flag in `_parse_args` (insert before `return parser.parse_args()`):

```python
    parser.add_argument(
        "--video-font-path",
        default=None,
        type=Path,
        help="Path to a TrueType font for tile labels. "
             "If omitted, auto-discover CJK → Latin system fonts, "
             "or honor $RHOS_COBOT_VIDEO_FONT.",
    )
```

4c. Add the helper near the other `_decode_frame` / `_prepare_frame_for_output` helpers:

```python
def _annotate_tile(
    tile: np.ndarray,
    label: str,
    font: ImageFont.FreeTypeFont,
    *,
    tile_width: int,
) -> np.ndarray:
    """Paint a black header strip + white label text on a contact-sheet tile.

    Returns a new BGR ndarray of the same shape as ``tile``.
    """
    pil_tile = bgr_to_pil(tile)
    from PIL import ImageDraw  # local import keeps top of file tidy

    draw = ImageDraw.Draw(pil_tile)
    draw_text_box(
        draw,
        (8, 6),
        label,
        font,
        padding=(6, 4),
        fg=(255, 255, 255),
        bg=(0, 0, 0),
        box_width=tile_width,
    )
    return pil_to_bgr(pil_tile)
```

4d. In `main()`, resolve font once before the sampling loop (insert right after `contact_path = ...`):

```python
    font_path = resolve_font_path(args.video_font_path)
    tile_font = load_font(18, font_path)
```

4e. Replace the existing tile labeling block (the `cv2.rectangle(...)` + `cv2.putText(...)` calls inside the sampling loop). Find this block:

```python
            label = f"frame {frame_idx}  t={frame_idx / fps:.1f}s"
            cv2.rectangle(frame, (0, 0), (args.tile_width, 28), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(frame)
```

Replace with:

```python
            label = f"frame {frame_idx}  t={frame_idx / fps:.1f}s"
            frame = _annotate_tile(frame, label, tile_font, tile_width=args.tile_width)
            tiles.append(frame)
```

- [ ] **Step 5: Run the new test to confirm it passes**

Run: `pytest tests/test_extract_replay_camera_video.py::test_annotate_tile_writes_label_with_black_background -v`
Expected: PASS

- [ ] **Step 6: Run full existing test file to check for regressions**

Run: `pytest tests/test_extract_replay_camera_video.py -v`
Expected: all tests PASS (new test + prior tests)

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_replay_camera_video.py tests/test_extract_replay_camera_video.py
git commit -m "refactor(extract_video): use pillow_overlay for contact-sheet labels"
```

---

### Task 9: Migrate `scripts/visualize_replay_progress.py`

Biggest migration: `compose_frame` builds a multi-panel canvas (status column + progress plot). Canvas fill + 1-px borders stay in cv2; text + polyline + markers + plot labels move to Pillow. CLI gains `--video-font-path`.

**Files:**
- Modify: `scripts/visualize_replay_progress.py`
- Modify: `tests/test_visualize_replay_progress.py`

- [ ] **Step 1: Read existing `compose_frame` implementation**

Run: `sed -n '263,391p' scripts/visualize_replay_progress.py`

You should see:
- `_write_status_lines` helper using `cv2.putText`
- `compose_frame` that:
  - builds a black canvas larger than the input frame
  - paints status panel `(22, 22, 22)` and progress panel `(14, 14, 14)` fills
  - draws 1-px `cv2.rectangle` borders
  - writes "progress history" label with `cv2.putText`
  - draws threshold line with `cv2.line` + label with `cv2.putText`
  - plots polyline with `cv2.line` and markers with `cv2.circle`

- [ ] **Step 2: Write new failing test covering font plumbing**

Append to `tests/test_visualize_replay_progress.py` (anywhere near the other `compose_frame` tests):

```python
def test_compose_frame_accepts_custom_font_path_via_env(tmp_path: Path, monkeypatch):
    """compose_frame must honor $RHOS_COBOT_VIDEO_FONT when no explicit font is passed."""
    import numpy as np

    from scripts.visualize_replay_progress import StepRecord, compose_frame

    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not dejavu.is_file():
        pytest.skip("DejaVuSans not available on this host")
    monkeypatch.setenv("RHOS_COBOT_VIDEO_FONT", str(dejavu))

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    records = [
        StepRecord(
            step=0, prompt="p", progress=0.4, progress_event="continue",
            trigger_reason="", replanner_called=False, replanner_action="",
            replanner_reason="", completed=False, camera_name="cam_high",
        )
    ]

    canvas = compose_frame(
        frame=frame, records=records, current_index=0, complete_threshold=0.85,
    )
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert np.any(canvas != 0)  # something got drawn
```

- [ ] **Step 3: Run new test to verify it fails as expected**

Run: `pytest tests/test_visualize_replay_progress.py::test_compose_frame_accepts_custom_font_path_via_env -v`
Expected: this may already PASS if no import error (since signature unchanged). Either way, continue — the real coverage below is that ALL 3 existing compose_frame tests continue to pass after the migration.

- [ ] **Step 4: Add `video_font_path` plumbing to Args / Config / CLI**

Edit `scripts/visualize_replay_progress.py`:

4a. Update imports (add near existing imports):

```python
from rhos_cobot.pillow_overlay import (
    bgr_to_pil,
    composite_overlay_on_bgr,  # imported for future use; not strictly needed here
    draw_marker,
    draw_polyline,
    draw_text_box,
    load_font,
    pil_to_bgr,
    resolve_font_path,
)
```

Note: `composite_overlay_on_bgr` is only needed if we want translucency. The status/plot panels here are opaque, so we draw directly on the RGB PIL image. We can omit it — include only what's used. Revised imports:

```python
from rhos_cobot.pillow_overlay import (
    bgr_to_pil,
    draw_marker,
    draw_polyline,
    draw_text_box,
    load_font,
    pil_to_bgr,
    resolve_font_path,
)
```

4b. Extend the `Args` dataclass (find the class, add a new field at the end of the class definition):

```python
@dataclass(frozen=True)
class Args:
    dataset_path: str
    checkpoint_dir: str
    output_video: str
    prompt: str
    start_step: int | None = None
    end_step: int | None = None
    dump_jsonl: str | None = None
    camera_name: str = "cam_high"
    task_decompose: bool = False
    no_replanner: bool = False
    video_font_path: Path | None = None   # NEW
```

4c. Extend `ReplayProgressVisualizationConfig` similarly — add `video_font_path: Path | None = None` at the end.

4d. Add CLI flag in `build_parser` (before `return parser`):

```python
    parser.add_argument(
        "--video-font-path",
        default=None,
        type=Path,
        help="Path to a TrueType font for overlay text. "
             "Default: auto-discover CJK → Latin, or honor $RHOS_COBOT_VIDEO_FONT.",
    )
```

4e. Plumb through in `parse_args` (after the existing return-value construction):

```python
    return Args(
        dataset_path=ns.dataset_path,
        checkpoint_dir=ns.checkpoint_dir,
        output_video=output_video,
        prompt=ns.prompt,
        start_step=ns.start_step,
        end_step=ns.end_step,
        dump_jsonl=dump_jsonl,
        camera_name=ns.camera_name,
        task_decompose=ns.task_decompose,
        no_replanner=ns.no_replanner,
        video_font_path=ns.video_font_path,   # NEW
    )
```

4f. Plumb through in `build_config`:

```python
    return ReplayProgressVisualizationConfig(
        dataset_path=Path(args.dataset_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_video=Path(args.output_video),
        prompt=args.prompt,
        start_step=args.start_step,
        end_step=args.end_step,
        dump_jsonl=Path(args.dump_jsonl) if args.dump_jsonl else None,
        camera_name=args.camera_name,
        task_decompose=args.task_decompose,
        no_replanner=args.no_replanner,
        video_font_path=args.video_font_path,   # NEW
    )
```

- [ ] **Step 5: Migrate `_write_status_lines` to Pillow**

Replace the existing `_write_status_lines` function body. The new version takes a PIL image + fonts instead of an ndarray.

Replace (find this function):

```python
def _write_status_lines(
    canvas: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    width: int,
    record: StepRecord,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    ...
```

With:

```python
def _write_status_lines(
    canvas_pil: Image.Image,
    *,
    origin_x: int,
    origin_y: int,
    width: int,
    record: StepRecord,
    font_body,
    font_small,
) -> None:
    """Paint the status column on a PIL RGB canvas in-place."""
    draw = ImageDraw.Draw(canvas_pil)
    color = (245, 245, 245)
    small_color = (210, 210, 210)
    y = origin_y + 28
    line_step = 28

    def _draw(label: str, value: str, *, use_small: bool = False) -> None:
        nonlocal y
        text = f"{label}: {value}"
        max_chars = max(12, int(width / 12))
        for line in textwrap.wrap(text, width=max_chars) or [""]:
            draw_text_box(
                draw,
                (origin_x + 12, y),
                line,
                font_small if use_small else font_body,
                padding=(0, 0),
                fg=small_color if use_small else color,
            )
            y += line_step

    _draw("prompt", record.prompt)
    _draw("step", str(record.step))
    _draw("progress", f"{record.progress:.3f}")
    _draw("event", record.progress_event)
    _draw("trigger", record.trigger_reason or "-", use_small=True)
    replanner_value = (
        f"{record.replanner_action or '-'} | {record.replanner_reason or '-'}"
        if record.replanner_called else "not called"
    )
    _draw("replanner", replanner_value, use_small=True)
```

Add `from PIL import Image, ImageDraw` near the existing `cv2`/`numpy` imports.

- [ ] **Step 6: Migrate `compose_frame`**

Replace the full `compose_frame` function body. The new version keeps cv2 for opaque panel fills and borders, but does all text + polyline + markers + threshold line on a single PIL pass.

```python
def compose_frame(
    frame: np.ndarray,
    records: Sequence[StepRecord],
    current_index: int,
    complete_threshold: float,
    *,
    font_body: ImageFont.FreeTypeFont | None = None,
    font_small: ImageFont.FreeTypeFont | None = None,
    video_font_path: Path | None = None,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape (H, W, 3)")
    if not records:
        raise ValueError("records must not be empty")

    current_index = _clamp_index(current_index, len(records))
    current = records[current_index]

    frame_h, frame_w = frame.shape[:2]
    status_width = max(320, frame_w // 2)
    progress_height = max(170, frame_h // 2)
    canvas_h = frame_h + progress_height
    canvas_w = frame_w + status_width

    # Build opaque panels + borders with cv2 (unchanged).
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=frame.dtype)
    canvas[:frame_h, :frame_w] = frame
    if status_width > 0:
        canvas[:frame_h, frame_w:] = (22, 22, 22)
    if progress_height > 0:
        canvas[frame_h:, :] = (14, 14, 14)
    if frame_w > 2 and frame_h > 2:
        cv2.rectangle(canvas, (1, 1), (frame_w - 2, frame_h - 2), (60, 60, 60), 1)
    cv2.rectangle(canvas, (frame_w, 0), (canvas_w - 1, frame_h - 1), (70, 70, 70), 1)
    cv2.rectangle(canvas, (0, frame_h), (canvas_w - 1, canvas_h - 1), (70, 70, 70), 1)

    # Resolve fonts (cached via lru_cache; env/args-agnostic default).
    if font_body is None or font_small is None:
        font_path = resolve_font_path(video_font_path)
        if font_body is None:
            font_body = load_font(16, font_path)
        if font_small is None:
            font_small = load_font(13, font_path)

    # Switch to PIL for all text + lines + markers.
    canvas_pil = bgr_to_pil(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    _write_status_lines(
        canvas_pil,
        origin_x=frame_w,
        origin_y=0,
        width=status_width,
        record=current,
        font_body=font_body,
        font_small=font_small,
    )

    plot_x0 = 18
    plot_y0 = frame_h + 24
    plot_w = canvas_w - 36
    plot_h = progress_height - 48
    plot_x1 = plot_x0 + plot_w
    plot_y1 = plot_y0 + plot_h

    draw_text_box(
        draw, (plot_x0, frame_h + 18),
        "progress history", font_body,
        padding=(0, 0), fg=(225, 225, 225),
    )

    threshold_y = plot_y0 + int(round((1.0 - complete_threshold) * max(plot_h - 1, 1)))
    draw.line([(plot_x0, threshold_y), (plot_x1, threshold_y)],
              fill=(0, 220, 220), width=1)
    draw_text_box(
        draw, (plot_x0 + 6, max(plot_y0 - 22, 2)),
        f"complete >= {complete_threshold:.2f}", font_small,
        padding=(0, 0), fg=(0, 220, 220),
    )

    history = list(records[: current_index + 1])
    if len(history) == 1:
        points = [(plot_x0 + plot_w // 2,
                   plot_y0 + int((1.0 - history[0].progress) * max(plot_h - 1, 1)))]
    else:
        points = []
        for idx, record in enumerate(history):
            x = plot_x0 + int(round(idx * plot_w / max(len(history) - 1, 1)))
            y = plot_y0 + int(round((1.0 - record.progress) * max(plot_h - 1, 1)))
            points.append((x, y))

    draw_polyline(draw, points, color=(90, 180, 90), width=2)

    for idx, record in enumerate(history):
        draw_marker(draw, points[idx], color=_marker_color(record), radius=4)

    draw_marker(draw, points[-1], color=(255, 255, 255), radius=7,
                outline=(255, 255, 255), outline_width=1)
    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1),
                   outline=(100, 100, 100), width=1)

    return pil_to_bgr(canvas_pil)
```

Note: `_marker_color` returns a BGR tuple in the original code (for cv2). Since we're now drawing on an RGB PIL image, we need to reverse these. Update `_marker_color`:

Find:

```python
def _marker_color(record: StepRecord) -> tuple[int, int, int]:
    if record.progress_event == "complete" or record.completed:
        return (80, 220, 80)
    if record.progress_event == "stall":
        return (0, 180, 255)
    if record.progress_event == "regression":
        return (60, 60, 255)
    if record.replanner_called:
        if record.replanner_action == "complete":
            return (255, 220, 80)
        if record.replanner_action == "continue":
            return (255, 170, 0)
        if record.replanner_action == "error":
            return (255, 80, 255)
    return (160, 220, 160)
```

These were BGR values. The intent per the old cv2 calls was:
- complete = green
- stall = orange-ish
- regression = red
- replanner complete = light blue/cyan
- replanner continue = blue/orange
- replanner error = magenta

To preserve visual appearance when we switch to RGB PIL, swap each tuple. Replace with:

```python
def _marker_color(record: StepRecord) -> tuple[int, int, int]:
    # RGB (for Pillow). Colors mirror the previous cv2-BGR palette.
    if record.progress_event == "complete" or record.completed:
        return (80, 220, 80)       # green (R,G,B) ≈ old BGR (80,220,80)
    if record.progress_event == "stall":
        return (255, 180, 0)       # orange (was BGR (0,180,255))
    if record.progress_event == "regression":
        return (255, 60, 60)       # red (was BGR (60,60,255))
    if record.replanner_called:
        if record.replanner_action == "complete":
            return (80, 220, 255)  # light blue/cyan (was BGR (255,220,80))
        if record.replanner_action == "continue":
            return (0, 170, 255)   # blue/orange (was BGR (255,170,0))
        if record.replanner_action == "error":
            return (255, 80, 255)  # magenta (symmetric)
    return (160, 220, 160)          # pale green
```

Also update the existing `test_compose_frame_preserves_camera_frame_in_top_left` expectation: the camera frame copied into the top-left is **BGR** `(13, 29, 47)`. `compose_frame` still returns BGR via `pil_to_bgr`, so the assertion `assert tuple(canvas[0, 0]) == (13, 29, 47)` remains correct. No test change needed.

- [ ] **Step 7: Plumb font resolution into `render_replay_visualization` so the font is loaded once per run (not per frame)**

Find `render_replay_visualization(runtime, *, output_video, dump_jsonl=None, fps=25.0)`. After the existing `records = run_replay_visualization(...)` call and before the `first_frame = get_image(...)` call, add:

```python
        font_path = resolve_font_path(runtime.run_config_font_path_or_none())
        font_body = load_font(16, font_path)
        font_small = load_font(13, font_path)
```

Wait — `ReplayRunConfig` doesn't carry `video_font_path` today. Simpler: thread it via `render_replay_visualization` signature. Actual edit:

Change the function signature:

```python
def render_replay_visualization(
    runtime: ReplayVisualizationRuntime,
    *,
    output_video: Path,
    dump_jsonl: Path | None = None,
    fps: float = 25.0,
    video_font_path: Path | None = None,
) -> list[StepRecord]:
```

Inside the function body, resolve fonts once before the compose loop:

```python
        font_path = resolve_font_path(video_font_path)
        font_body = load_font(16, font_path)
        font_small = load_font(13, font_path)
```

Pass them to `compose_frame`:

```python
        first_canvas = compose_frame(
            first_frame,
            records,
            0,
            runtime.run_config.complete_threshold,
            font_body=font_body,
            font_small=font_small,
        )
        ...
        video_sink.write(
            compose_frame(
                frame,
                records,
                index,
                runtime.run_config.complete_threshold,
                font_body=font_body,
                font_small=font_small,
            )
        )
```

Update the call site in `main()`:

```python
        render_replay_visualization(
            runtime,
            output_video=config.output_video,
            dump_jsonl=config.dump_jsonl,
            video_font_path=config.video_font_path,   # NEW
        )
```

- [ ] **Step 8: Run all tests for this file**

Run: `pytest tests/test_visualize_replay_progress.py -v`
Expected: all tests PASS (existing + the new `test_compose_frame_accepts_custom_font_path_via_env`)

If the 3 existing `test_compose_frame_*` tests fail due to font resolution raising, verify DejaVuSans is on the host (`ls /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`). On CI where no fonts exist, set `RHOS_COBOT_VIDEO_FONT` via conftest or a pytest fixture in the existing test file. For now, assume DejaVuSans is present (already verified in the exploration phase).

- [ ] **Step 9: Commit**

```bash
git add scripts/visualize_replay_progress.py tests/test_visualize_replay_progress.py
git commit -m "refactor(visualize_replay_progress): migrate compose_frame overlay to pillow"
```

---

### Task 10: Migrate `examples/piper_real/replay_visualizer.py`

Live recording HUD. Requires degraded-path handling: if font resolution fails, disable HUD for recording (web UI still works) and emit a prominent warning.

**Files:**
- Modify: `examples/piper_real/replay_visualizer.py`
- Modify: `tests/` — add `tests/test_replay_visualizer.py`

- [ ] **Step 1: Write failing tests for constructor + degraded path**

Create new file `tests/test_replay_visualizer.py`:

```python
"""Tests for examples.piper_real.replay_visualizer.ReplayVisualizer overlay."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _install_cv2_stub(monkeypatch) -> None:
    """Minimal cv2 stub covering the calls replay_visualizer makes."""
    cv2_stub = types.SimpleNamespace()
    cv2_stub.COLOR_RGB2BGR = object()
    cv2_stub.INTER_AREA = 0

    def cvt_color(frame: np.ndarray, code: object) -> np.ndarray:
        return frame[..., ::-1].copy()

    def resize(frame, size, interpolation):
        return frame  # identity for tests that only care about overlay

    def imencode(ext, frame, params):
        return True, np.zeros(10, dtype=np.uint8)

    class _VideoWriter:
        def __init__(self, *args, **kwargs):
            self._opened = False
            self.frames: list[np.ndarray] = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            pass

    def video_writer_fourcc(*_a):
        return 0

    cv2_stub.cvtColor = cvt_color
    cv2_stub.resize = resize
    cv2_stub.imencode = imencode
    cv2_stub.IMWRITE_JPEG_QUALITY = 0
    cv2_stub.VideoWriter = _VideoWriter
    cv2_stub.VideoWriter_fourcc = video_writer_fourcc
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)


def _fake_env(monkeypatch):
    """Build a minimal ReplayEnvironment-shaped stand-in."""
    env = types.SimpleNamespace()
    env.camera_names = ("cam_high",)
    env.num_steps = 5
    env.fps = 25.0

    def get_image(name: str, step: int) -> np.ndarray:
        return np.full((60, 80, 3), 120, dtype=np.uint8)  # RGB mid-grey

    env.get_image = get_image
    return env


def _load_visualizer_module(monkeypatch):
    _install_cv2_stub(monkeypatch)
    from examples.piper_real import replay_visualizer as rv
    return rv


def test_constructor_resolves_font_when_available(tmp_path, monkeypatch):
    rv = _load_visualizer_module(monkeypatch)
    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not dejavu.is_file():
        pytest.skip("no system font available")

    vis = rv.ReplayVisualizer(
        _fake_env(monkeypatch),
        enabled=False,
        save_path="",
        video_font_path=dejavu,
    )
    assert vis._font_title is not None
    assert vis._font_body is not None
    vis.close()


def test_constructor_degrades_when_font_missing(monkeypatch, caplog, tmp_path):
    """Font-missing must emit a loud warning and disable HUD (non-fatal)."""
    rv = _load_visualizer_module(monkeypatch)
    from rhos_cobot import pillow_overlay as po

    # Force resolve_font_path to fail regardless of host env.
    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(tmp_path / "absent.ttc"),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(tmp_path / "absent.ttf"),))

    with caplog.at_level("WARNING"):
        vis = rv.ReplayVisualizer(_fake_env(monkeypatch), enabled=False, save_path="")

    assert vis._font_title is None
    assert vis._font_body is None
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "!!!" in joined
    assert "HUD" in joined
    vis.close()


def test_compose_record_frame_returns_plain_canvas_when_font_missing(monkeypatch, tmp_path):
    rv = _load_visualizer_module(monkeypatch)
    from rhos_cobot import pillow_overlay as po

    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(tmp_path / "x.ttc"),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(tmp_path / "y.ttf"),))

    vis = rv.ReplayVisualizer(_fake_env(monkeypatch), enabled=False, save_path="")
    frame_bgr = np.full((60, 80, 3), 120, dtype=np.uint8)
    canvas = vis._compose_record_frame(0, {"cam_high": frame_bgr}, extra_info="")

    # With no font, no HUD burned in → canvas is exactly the tile (or a tile mosaic).
    assert canvas.shape == frame_bgr.shape
    assert np.array_equal(canvas, frame_bgr)
    vis.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_replay_visualizer.py -v`
Expected: tests FAIL (`TypeError: __init__() got an unexpected keyword argument 'video_font_path'` or similar — the constructor doesn't accept the new kwarg yet).

- [ ] **Step 3: Update imports in replay_visualizer**

Edit `examples/piper_real/replay_visualizer.py` imports block (at the top of the file, after `import logging`):

```python
from pathlib import Path

from rhos_cobot.pillow_overlay import (
    FontUnavailableError,
    _FONT_ENV_VAR,
    bgr_to_pil,
    composite_overlay_on_bgr,
    draw_text_box,
    load_font,
    max_text_width,
    new_overlay,
    resolve_font_path,
)
```

- [ ] **Step 4: Update `__init__` to accept `video_font_path` and resolve font with degraded fallback**

Change the constructor signature in `ReplayVisualizer.__init__` to add the new keyword:

```python
    def __init__(
        self,
        environment: "ReplayEnvironment",
        *,
        enabled: bool = True,
        port: int = _DEFAULT_PORT,
        save_path: str = "",
        video_font_path: Path | None = None,
    ) -> None:
```

Right after the existing `self._video_fps = float(...)` initialization line (before the `self._lock = threading.Lock()` line), insert the font-resolution block:

```python
        # Font for burned-in HUD on recorded videos. Missing font is non-fatal:
        # the live web UI and the raw recording both continue to work — we
        # just skip the HUD overlay in the saved video.
        self._font_title = None
        self._font_body = None
        try:
            font_path = resolve_font_path(video_font_path)
            self._font_title = load_font(18, font_path)
            self._font_body = load_font(14, font_path)
        except FontUnavailableError as exc:
            logging.warning(
                "!!! Replay visualizer: no usable font found — HUD overlay "
                "is DISABLED for the recorded video (web UI is unaffected). "
                "To enable HUD: install fonts-noto-cjk, set $%s, or pass "
                "video_font_path=. (%s)",
                _FONT_ENV_VAR, exc,
            )
```

- [ ] **Step 5: Delete three obsolete cv2-based helpers**

Delete these methods from `ReplayVisualizer` in their entirety:
- `_get_uniform_overlay_box_width`
- `_draw_translucent_rect`
- `_draw_text_with_background`

- [ ] **Step 6: Rewrite `_compose_record_frame`**

Replace the full body of `_compose_record_frame` with:

```python
    def _compose_record_frame(
        self,
        step_idx: int,
        bgr_frames: dict[str, np.ndarray],
        extra_info: str,
    ) -> np.ndarray:
        with self._lock:
            subtask_idx = self._subtask_idx
            total_subtasks = self._total_subtasks
            subtask_type = self._subtask_type
            subtask_prompt = self._subtask_prompt

        ordered = [bgr_frames[name] for name in self._cam_names if name in bgr_frames]
        if not ordered:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        tile_h = max(frame.shape[0] for frame in ordered)
        tile_w = max(frame.shape[1] for frame in ordered)
        normalized: list[np.ndarray] = []
        for frame in ordered:
            if frame.shape[0] != tile_h or frame.shape[1] != tile_w:
                frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            normalized.append(frame)

        cols = min(3, len(normalized)) if len(normalized) > 1 else 1
        rows = (len(normalized) + cols - 1) // cols
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for idx, frame in enumerate(normalized):
            r = idx // cols
            c = idx % cols
            canvas[r * tile_h: (r + 1) * tile_h, c * tile_w: (c + 1) * tile_w] = frame

        # If no font is available, skip the HUD overlay entirely.
        if self._font_title is None or self._font_body is None:
            return canvas

        title = f"Replay step {step_idx + 1}/{self._total_steps}"
        subtask = f"Subtask {subtask_idx}/{total_subtasks} [{subtask_type}]"
        prompt_line = f"Prompt: {subtask_prompt}"[:200]
        info_line = f"Info: {extra_info}"[:200] if extra_info else ""

        overlay, draw = new_overlay((canvas.shape[1], canvas.shape[0]))
        candidate_texts = [title, subtask, prompt_line]
        if info_line:
            candidate_texts.append(info_line)
        box_width = max_text_width(draw, self._font_body, candidate_texts, padding_x=8)
        box_width = max(box_width, 220)
        box_width = min(box_width, max(220, canvas.shape[1] - 24))

        draw_text_box(
            draw, (16, 22), title, self._font_title,
            padding=(8, 5), fg=(245, 245, 245, 255),
            bg=(96, 96, 96, 122), box_width=box_width,
        )
        draw_text_box(
            draw, (16, 52), subtask, self._font_body,
            padding=(8, 4), fg=(185, 228, 255, 255),
            bg=(96, 96, 96, 122), box_width=box_width,
        )
        draw_text_box(
            draw, (16, 80), prompt_line, self._font_body,
            padding=(8, 5), fg=(224, 224, 224, 255),
            bg=(96, 96, 96, 118), box_width=box_width,
        )
        if info_line:
            draw_text_box(
                draw, (16, 108), info_line, self._font_body,
                padding=(8, 5), fg=(166, 210, 255, 255),
                bg=(96, 96, 96, 118), box_width=box_width,
            )

        return composite_overlay_on_bgr(canvas, overlay)
```

Note: `composite_overlay_on_bgr` expects a BGR frame in. `canvas` here is already BGR (built from `bgr_frames` values, which the caller converted with `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` in `update()`). So the call is correct.

- [ ] **Step 7: Run all related tests**

Run: `pytest tests/test_replay_visualizer.py -v`
Expected: all 3 new tests PASS.

- [ ] **Step 8: Commit**

```bash
git add examples/piper_real/replay_visualizer.py tests/test_replay_visualizer.py
git commit -m "refactor(replay_visualizer): migrate HUD overlay to pillow with degraded fallback"
```

---

## Phase C — Full regression pass

### Task 11: Run full test suite + smoke-test CLI entrypoints

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS. Note any unexpected failures — most tests here are isolated by cv2/h5py stubs and should be unaffected.

- [ ] **Step 2: Smoke-test the pillow_overlay module at the REPL**

Run:

```bash
python - <<'EOF'
from rhos_cobot.pillow_overlay import (
    resolve_font_path, load_font, new_overlay, draw_text_box, composite_overlay_on_bgr,
)
import numpy as np
font_path = resolve_font_path()
print(f"font: {font_path}")
font = load_font(18, font_path)
frame = np.full((100, 300, 3), 80, dtype=np.uint8)  # BGR grey
overlay, draw = new_overlay((frame.shape[1], frame.shape[0]))
draw_text_box(draw, (20, 30), "Hello 你好", font, fg=(255,255,255,255), bg=(0,0,0,160), padding=(8,5))
result = composite_overlay_on_bgr(frame, overlay)
print(f"result shape: {result.shape}, dtype: {result.dtype}")
print(f"pixels differ: {(result != frame).any()}")
EOF
```

Expected:
- Prints `font: /usr/share/fonts/...` (one of the candidates)
- Prints `result shape: (100, 300, 3), dtype: uint8`
- Prints `pixels differ: True`
- No tracebacks

- [ ] **Step 3: Smoke-test `--video-font-path` CLI parsing**

Run: `python -m scripts.extract_replay_camera_video --help 2>&1 | grep -A1 "video-font-path"`
Expected: shows the `--video-font-path` flag with help text.

Run: `python -m scripts.visualize_replay_progress --help 2>&1 | grep -A1 "video-font-path"`
Expected: shows the `--video-font-path` flag with help text.

- [ ] **Step 4: Commit (only if any whitespace / import fixes surfaced)**

If Steps 1–3 surfaced nothing, skip this step. If you had to fix something (e.g., an unused import, a stray typo), commit it:

```bash
git add -p
git commit -m "chore(overlay): post-migration cleanup"
```

---

## Done

End state: 4 commits on `feat-gui` (plus the spec commit from the brainstorming phase):

1. `feat(overlay): scaffold rhos_cobot.pillow_overlay module`
2. `feat(overlay): add resolve_font_path with CJK/Latin/env fallbacks`
3. `feat(overlay): add lru-cached load_font`
4. `feat(overlay): add bgr_to_pil / pil_to_bgr conversion helpers`
5. `feat(overlay): add draw_text_box + max_text_width`
6. `feat(overlay): add draw_polyline + draw_marker`
7. `feat(overlay): add new_overlay + composite_overlay_on_bgr`
8. `refactor(extract_video): use pillow_overlay for contact-sheet labels`
9. `refactor(visualize_replay_progress): migrate compose_frame overlay to pillow`
10. `refactor(replay_visualizer): migrate HUD overlay to pillow with degraded fallback`
11. (optional) `chore(overlay): post-migration cleanup`

`main` branch is untouched; only `feat-gui`-added files changed. Pillow was already in `examples/piper_real/requirements.in`/`.txt` — no dependency edits required.
