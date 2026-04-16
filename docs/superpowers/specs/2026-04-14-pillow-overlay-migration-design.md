# Pillow Overlay Migration — Design Spec

**Date:** 2026-04-14
**Branch:** `feat-gui`
**Status:** approved — ready for implementation planning

## 1. Overview

Replace `cv2.putText` / `cv2.line` / `cv2.circle` / `cv2.rectangle` / `cv2.addWeighted`
based overlay drawing in three feat-gui files with Pillow (`PIL.ImageDraw` + TrueType
fonts). The goal is higher-quality text rendering (hinting + anti-aliasing), native
CJK support, and visually consistent anti-aliased curves/markers on progress plots.

No video encoder change: `cv2.VideoWriter(mp4v)` stays. Scope limited to overlay
pixels only.

## 2. Motivation

`cv2.putText` uses the coarse Hershey vector font family with no hinting, no
subpixel anti-aliasing, and no CJK support. Current demo/debug videos render
Chinese subtask prompts as `▢▢▢` and Latin text looks pixelated at small sizes.
`cv2.line` / `cv2.circle` with `LINE_AA` also look noticeably rougher than Pillow's
equivalents when videos are viewed at native resolution.

## 3. Scope

### In scope (3 feat-gui-added files)
- `examples/piper_real/replay_visualizer.py` — live multi-camera recording HUD
- `scripts/visualize_replay_progress.py` — offline progress-visualizer canvas
- `scripts/extract_replay_camera_video.py` — contact-sheet tile labels

### New files
- `rhos_cobot/pillow_overlay.py` — shared overlay helper module
- `tests/test_pillow_overlay.py` — unit tests

### Out of scope
- `rhos_cobot/post_process.py` (exists on `main`; not touched)
- `examples/piper_real/main.py` (exists on `main`; not touched)
- Video encoder / codec change (future work: `imageio-ffmpeg` + `libx264`)
- Progress curve layout changes (visual style kept the same, only drawing
  backend swapped)
- `cv2.VideoWriter` and the multi-camera tile stitching (opaque solid fills
  stay in cv2 — no画质 benefit from moving them)

## 4. Architecture

```
rhos_cobot/
├── pillow_overlay.py          (new — shared)
└── post_process.py            (unchanged)

examples/piper_real/
└── replay_visualizer.py       (modified — imports pillow_overlay)

scripts/
├── visualize_replay_progress.py   (modified)
└── extract_replay_camera_video.py (modified)

tests/
└── test_pillow_overlay.py     (new)
```

All three consumer files import `from rhos_cobot.pillow_overlay import ...`.
This keeps the helper at the top-level package level (no cross-package
back-imports from `scripts/` → `examples/`) and co-locates it with other
existing `rhos_cobot` utilities (`post_process.py`, `utils.py`).

## 5. Module API: `rhos_cobot/pillow_overlay.py`

### 5.1 Errors

```python
class FontUnavailableError(RuntimeError):
    """Raised when no usable font can be resolved.

    Message includes the three remediation paths:
      - install a CJK font (e.g., `apt install fonts-noto-cjk`)
      - set environment variable $RHOS_COBOT_VIDEO_FONT
      - pass --video-font-path on the CLI
    """
```

### 5.2 Font discovery

```python
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
    """Resolve a font file path with this priority:
        1. user_path (if provided, must exist)
        2. $RHOS_COBOT_VIDEO_FONT (if set, must exist)
        3. _CJK_FONT_CANDIDATES (first existing)
        4. _LATIN_FONT_CANDIDATES (first existing)
        5. raise FontUnavailableError
    """
```

Rationale: CJK tried before Latin so Chinese prompts render correctly
by default; Latin fallback is present but is an active degradation, not
a silent win — if the caller expected CJK and only Latin was found, text
may still render as `▢` and that will be visible to the user.

### 5.3 Font loading

```python
@functools.lru_cache(maxsize=32)
def load_font(size: int, font_path: Path) -> ImageFont.FreeTypeFont:
    """Load (and cache) a TrueType font at a given size."""
```

Cache size of 32 covers the expected usage (≤3 size buckets × 1–2 font
paths × modest headroom).

### 5.4 Array / image conversion

```python
def bgr_to_pil(frame: np.ndarray) -> Image.Image
def pil_to_bgr(image: Image.Image) -> np.ndarray
```

Both validate shape `(H, W, 3)` with H > 0 and W > 0 and `dtype=uint8`;
raise `ValueError` with actionable message otherwise. `bgr_to_pil` returns
an `Image` in `"RGB"` mode.

### 5.5 Drawing primitives (direct-on-RGB, no alpha)

```python
_RGB = tuple[int, int, int]
_RGBA = tuple[int, int, int, int]

def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    padding: tuple[int, int] = (8, 5),
    fg: _RGB | _RGBA = (245, 245, 245),
    bg: _RGB | _RGBA | None = None,
    radius: int = 0,
    box_width: int | None = None,
) -> tuple[int, int, int, int]
```
- Draws a filled rectangle (if `bg` is not None) at `(x-px, y-py, x-px+box_width, y+h+py)`
- Draws the text at `(x, y)` (origin = text top-left)
- Returns the full bbox `(x0, y0, x1, y1)` so callers can stack lines
- If `box_width` is None the box sizes to the text; otherwise forces uniform width

```python
def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    width: int = 2,
) -> None

def draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    color: tuple[int, int, int],
    radius: int,
    outline: tuple[int, int, int] | None = None,
    outline_width: int = 0,
) -> None
```
- `draw_polyline` uses `ImageDraw.line(..., joint='curve')` for anti-aliased joins
- `draw_marker` draws a filled circle via `ImageDraw.ellipse`, optional outline circle

### 5.6 Alpha-composite overlay (for translucent HUDs)

```python
def new_overlay(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Return (RGBA Image with alpha=0, ImageDraw on that image)."""

def composite_overlay_on_bgr(
    frame: np.ndarray,
    overlay: Image.Image,
) -> np.ndarray:
    """Alpha-composite an RGBA overlay onto a BGR ndarray frame.
    Returns a new BGR ndarray (does not mutate frame).
    """
```

Usage pattern (for translucent HUDs):
```python
h, w = canvas.shape[:2]
overlay, draw = new_overlay((w, h))
draw_text_box(draw, (16, 30), title, font_title,
              fg=(245,245,245), bg=(96,96,96,120), radius=6)
canvas = composite_overlay_on_bgr(canvas, overlay)
```

Note: inside overlay-mode, `draw_text_box` accepts an **RGBA** `bg` tuple
(4-tuple vs 3-tuple) to produce translucent backgrounds. Type is
`tuple[int,int,int] | tuple[int,int,int,int] | None`.

### 5.7 Layout helper

```python
def max_text_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    texts: Iterable[str],
    *,
    padding_x: int,
) -> int:
    """Return max(text_width + 2*padding_x) across texts."""
```
Used by `replay_visualizer` to keep HUD lines in a uniform-width column.

## 6. Per-file migration plan

### 6.1 `examples/piper_real/replay_visualizer.py`

Constructor changes:
```python
def __init__(
    self,
    environment: ReplayEnvironment,
    *,
    enabled: bool = True,
    port: int = _DEFAULT_PORT,
    save_path: str = "",
    video_font_path: Path | None = None,   # new
) -> None:
```

During `__init__`, resolve font and load two size buckets (title / body).
On `FontUnavailableError`:

```python
try:
    font_path = resolve_font_path(video_font_path)
    self._font_title = load_font(18, font_path)
    self._font_body = load_font(14, font_path)
except FontUnavailableError as exc:
    logging.warning(
        "!!! Replay visualizer: no usable font found — HUD overlay will "
        "be DISABLED for the recorded video. Web preview still works. "
        "To enable HUD: install fonts-noto-cjk, set $%s, or pass "
        "video_font_path. (%s)",
        _FONT_ENV_VAR, exc,
    )
    self._font_title = None
    self._font_body = None
```

When `_font_title is None` in `_compose_record_frame`, skip the overlay
block entirely and return the unadorned tile canvas (still records video,
still serves web UI — just no burned-in text).

Delete:
- `_draw_translucent_rect`
- `_draw_text_with_background`
- `_get_uniform_overlay_box_width`

`_compose_record_frame` new shape (pseudocode):
```python
canvas = <build multi-cam tile canvas as before with cv2>
if self._font_title is None:
    return canvas
overlay, draw = new_overlay((canvas.shape[1], canvas.shape[0]))
box_width = max_text_width(draw, self._font_body, [title, subtask, prompt, info], padding_x=8)
draw_text_box(draw, (16, 30), title, self._font_title,
              fg=(245,245,245), bg=(96,96,96,120), box_width=box_width)
draw_text_box(draw, (16, 58), subtask, self._font_body,
              fg=(185,228,255), bg=(96,96,96,120), box_width=box_width)
# ... etc
return composite_overlay_on_bgr(canvas, overlay)
```

Web-preview path (JSON state endpoint) is unchanged; it uses raw JPEGs,
not composited HUD.

### 6.2 `scripts/visualize_replay_progress.py`

CLI / config additions:
```python
parser.add_argument("--video-font-path", default=None, type=Path,
                    help="Path to a TrueType font for overlay text.")
```

Add `video_font_path: Path | None` field to `Args` and
`ReplayProgressVisualizationConfig`.

`compose_frame(frame, records, current_index, complete_threshold, *, font_body, font_small)`
— add two font params. Callers (`render_replay_visualization`) resolve
once via `resolve_font_path` + `load_font` before the per-frame loop,
then pass in.

On `FontUnavailableError` during resolution: **raise** (no degraded path
— this is an offline analysis tool where missing text defeats its
purpose; caller sees a clear error at CLI startup).

Migration inside `compose_frame`:
- Keep: cv2-built canvas background, solid fills for status panel / plot
  panel, 1-px cv2 border rectangles (opaque, cv2 renders fine)
- Replace:
  - `_write_status_lines` body — `cv2.putText` → `draw_text_box` (no bg
    fill, direct on RGB since panel is already dark solid)
  - `cv2.putText` for "progress history" label and threshold label →
    `draw_text_box`
  - threshold horizontal line (`cv2.line` single solid stroke) → swap to
    `ImageDraw.line` for consistency, since the frame is already on PIL
    when we get here
  - progress polyline (`cv2.line` loop) → `draw_polyline`
  - event markers (`cv2.circle` loop) → `draw_marker`
  - current-step highlight ring (`cv2.circle` outline) → `draw_marker`
    with `outline=` kwarg
- Flow: build base canvas with cv2 as before → `bgr_to_pil(canvas)` once
  → draw all text/curves/markers via `ImageDraw` → `pil_to_bgr` → return

### 6.3 `scripts/extract_replay_camera_video.py`

CLI addition: `--video-font-path` (same shape as above).

Contact-sheet tile labeling path only (the MP4 writing loop has no text
overlay and stays untouched):

```python
# Replace the old:
#   cv2.rectangle(frame, (0,0), (tile_width, 28), (0,0,0), -1)
#   cv2.putText(frame, label, (8,20), FONT_HERSHEY_SIMPLEX, 0.6, ...)

pil_tile = bgr_to_pil(tile)
draw = ImageDraw.Draw(pil_tile)
draw_text_box(draw, (8, 6), label, font,
              fg=(255,255,255), bg=(0,0,0), padding=(6,4), box_width=args.tile_width)
tile = pil_to_bgr(pil_tile)
```

Font resolved once before the sampling loop via `resolve_font_path` /
`load_font`. On `FontUnavailableError`: **raise** (same reasoning as
progress visualizer — offline tool, failing loud is correct).

## 7. Testing strategy: `tests/test_pillow_overlay.py`

Target: pure-Python tests, no cv2.VideoWriter / openpi / ROS / real HDF5.

| Test | What it validates |
|---|---|
| `test_resolve_font_user_path_exists` | Existing `tmp_path` .ttf → returned unchanged |
| `test_resolve_font_user_path_missing_raises` | Nonexistent path → `FileNotFoundError` |
| `test_resolve_font_env_var_used` | `monkeypatch.setenv(_FONT_ENV_VAR, str(existing))` → returned |
| `test_resolve_font_env_var_missing_raises` | Env var set but file missing → `FileNotFoundError` |
| `test_resolve_font_cjk_preferred` | monkeypatch candidates to (fake_cjk, fake_latin), both exist → CJK returned |
| `test_resolve_font_falls_back_to_latin` | monkeypatch CJK candidates empty, Latin present → Latin returned |
| `test_resolve_font_all_missing_raises` | Both lists empty + no env + no user path → `FontUnavailableError` with helpful message |
| `test_load_font_lru_cached` | Two calls with same `(size, path)` return the same object (`is`) |
| `test_bgr_to_pil_color_swap` | Input BGR (B=0,G=0,R=255) → PIL pixel (R=255,G=0,B=0) |
| `test_pil_to_bgr_roundtrip` | `pil_to_bgr(bgr_to_pil(frame))` bit-equal to `frame` |
| `test_bgr_to_pil_invalid_shape` | 2D ndarray → ValueError |
| `test_draw_text_box_returns_bbox` | Returned bbox dims match `draw.textbbox` plus padding |
| `test_draw_text_box_writes_pixels` | Pixels inside bbox differ from starting canvas |
| `test_draw_text_box_uniform_box_width` | With `box_width=200`, returned bbox width = 200 |
| `test_draw_polyline_renders` | Three-point polyline → pixels along each segment midpoint non-background |
| `test_draw_marker_fills_center` | Filled circle center pixel == color, outside radius+1 == background |
| `test_composite_overlay_blends_alpha` | Red overlay 50% alpha on green BGR frame → center pixel roughly (128, 128, 0) BGR; corners outside overlay unchanged |
| `test_max_text_width_returns_widest` | Three strings, widest determines output (± padding) |

Font used in drawing tests: `ImageFont.load_default()` to avoid a
system-font dependency in CI. That font is bitmap, not freetype, so we
cannot test hinting — but we can test pixel change + bbox correctness,
which is what the API promises.

For `test_resolve_font_*` the "fake" font paths are `tmp_path / "x.ttf"`
created with `.touch()` (empty file is fine; `resolve_font_path` only
checks existence — `load_font` is what actually loads bytes).

## 8. Error handling & edge cases

| Case | Behavior |
|---|---|
| Font not resolvable — `replay_visualizer` (live demo) | `logging.warning` with "!!!" marker, disable HUD, continue recording + web |
| Font not resolvable — `visualize_replay_progress` (offline) | raise, fail at CLI startup |
| Font not resolvable — `extract_replay_camera_video` (offline) | raise, fail at CLI startup |
| Empty or zero-dim frame into `bgr_to_pil` | `ValueError` |
| Non-uint8 frame into `bgr_to_pil` | `ValueError` |
| `box_width` smaller than text | Pillow clips text at box edge; document as "caller responsibility to size correctly" |
| Text with characters missing from font | Pillow renders `.notdef` glyph (□); acceptable — user's responsibility to supply appropriate font |

## 9. Performance notes

- `bgr_to_pil` / `pil_to_bgr` are zero-copy aside from channel swap
  (numpy slicing `[..., ::-1]`) — sub-millisecond on 1080p
- `Image.alpha_composite` on 1080p: ~2–3 ms on modern CPU
- Font rendering: dominated by number of glyphs; 4 HUD lines × ~40 chars
  ≈ 1 ms
- All well within the 40 ms / 25 fps budget for `replay_visualizer`'s
  live recording path
- Offline tools are not fps-bound

### 9.1 Known line rendering limitation

Pillow's `ImageDraw.line()` is **not** anti-aliased by default — only
text (via FreeType hinting) and filled shapes like `ellipse` / `rounded_rectangle`
are. For the progress curve in `visualize_replay_progress.py` this means
the switch from `cv2.line(..., LINE_AA)` to Pillow `line()` is roughly
visually **lateral**, not a clear upgrade, for thin `width=2` strokes.

Where Pillow clearly wins in this migration:
- All text (vs. cv2's Hershey font) — big win, CJK support
- Event markers (`cv2.circle` → `ImageDraw.ellipse`) — AA circles
- Rounded text boxes (`cv2.addWeighted` rect → `ImageDraw.rounded_rectangle`) — AA corners

Where Pillow is roughly tied or slightly worse:
- The progress polyline itself (aliased segments)

If polyline aliasing turns out to be visibly bad in review, the fix is a
2× supersample pass — implementation can add a `supersample: int = 1`
kwarg to `draw_polyline` that internally draws on a scaled RGBA image
and downsamples with `Image.LANCZOS`. This is **not** in V1 scope; add
only if reviewers dislike the aliased curve.

## 10. Dependencies

Pillow is added to the runtime dependency list. Already declared
transitively in several places, but we will explicitly add it to:
- `examples/piper_real/requirements.in` / `.txt`

Pytest and Pillow are the only new dev-time deps for tests; both are
already present.

## 11. Risks / open items

- **Live recording degraded path** (font missing → no HUD). Chosen over
  hard-failing the live demo because losing the demo recording is worse
  than losing burned-in text; web UI still shows all state. Warning is
  prefixed with "!!!" so it's visible in terminal logs.
- **CJK font presence on deployment machines** — owner should ensure
  `fonts-noto-cjk` or equivalent is installed on robot controllers; spec
  does **not** bundle a font in-repo (would add several MB to git).
- **Pillow version skew** — `ImageDraw.line(..., joint='curve')` requires
  Pillow ≥ 8.0 (we're on ≥10 in requirements already).

---

## Approval trail

- § 1 file list — approved
- § 2 public API shape — approved
- § 3 per-file changes — approved
- § 4 testing strategy — approved (test file shape agreed)
- § 5 error handling — approved (§5 degrade-with-warning chosen for
  `replay_visualizer`; hard-fail chosen for the two offline tools)
