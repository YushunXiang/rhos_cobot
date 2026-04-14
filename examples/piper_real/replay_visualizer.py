"""Web-based camera + subtask visualization for HDF5 replay.

Starts a lightweight HTTP server in a background thread.  Open the URL
in any browser to see tiled camera views with subtask overlay, updated
in real time as the replay progresses.  No extra dependencies required
(uses only Python stdlib + cv2 + numpy).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rhos_cobot.pillow_overlay import (
    FontUnavailableError,
    _FONT_ENV_VAR,
    composite_overlay_on_bgr,
    draw_text_box,
    load_font,
    max_text_width,
    new_overlay,
    resolve_font_path,
)

if TYPE_CHECKING:
    from examples.piper_real.replay_env import ReplayEnvironment

_DEFAULT_PORT = 7860

# ── HTML served to the browser ──────────────────────────────────────────────

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Replay Visualizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: #1a1a2e;
        color: #eee;
        font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', system-ui, sans-serif;
    }
    #stage {
        position: relative;
        padding: 12px;
    }
    #hud {
        position: absolute;
        top: 20px;
        left: 20px;
        z-index: 3;
        max-width: min(62vw, 960px);
        display: flex;
        flex-direction: column;
        gap: 6px;
        pointer-events: none;
    }
    .hud-line {
        background: rgba(88, 92, 104, 0.45);
        border: 1px solid rgba(180, 185, 198, 0.25);
        border-radius: 6px;
        padding: 4px 9px;
        backdrop-filter: blur(1px);
        text-shadow: 0 1px 1px rgba(0, 0, 0, 0.35);
        width: 100%;
    }
    #progress { font-size: 12px; font-weight: 600; color: #f4f5fb; }
    #subtask-index { font-size: 11px; color: #dde3f2; margin-right: 6px; }
    #subtask-type { display: inline-block; font-size: 11px; font-weight: 700; }
    #subtask-type.navigate { color: #8ff7e6; }
    #subtask-type.manipulate { color: #ffb6c3; }
    #subtask-type.policy { color: #c8c4ff; }
    #subtask-prompt { font-size: 16px; font-weight: 600; color: #f0f0f4; max-width: 100%; }
    #extra-info { font-size: 15px; font-weight: 600; color: #b8d3ff; }
  #cameras {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        width: 100%;
  }
    .cam-tile { position: relative; min-width: 0; }
    .cam-tile img {
        display: block;
        width: 100%;
        height: clamp(200px, 28vw, 340px);
        object-fit: contain;
        background: #0d1021;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
  .cam-label {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(90, 90, 100, 0.55);
        border: 1px solid rgba(186, 186, 196, 0.3);
        color: #f2f4f8;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        text-shadow: 0 1px 1px rgba(0, 0, 0, 0.45);
  }
  #status { text-align: center; padding: 8px; font-size: 12px; color: #555; }
    @media (max-width: 1100px) {
        #cameras { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
        #stage { padding: 8px; }
        #cameras { grid-template-columns: 1fr; }
        #hud {
            top: 14px;
            left: 14px;
            right: 14px;
            max-width: none;
        }
    }
</style>
</head>
<body>
    <div id="stage">
        <div id="cameras"></div>
        <div id="hud">
            <div class="hud-line" id="progress">Step -/-</div>
            <div class="hud-line">
                <span id="subtask-index">-/-</span>
                <span id="subtask-type">-</span>
            </div>
            <div class="hud-line" id="subtask-prompt">-</div>
            <div class="hud-line" id="extra-info"></div>
        </div>
  </div>
  <div id="status">connecting...</div>
<script>
const POLL_MS = 150;
let lastStep = -1;
async function poll() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) { setTimeout(poll, POLL_MS); return; }
    const d = await r.json();
    if (d.done) {
      document.getElementById('status').textContent = 'Replay finished.';
      return;
    }
    document.getElementById('progress').textContent =
      'Step ' + d.step + '/' + d.total_steps;
    document.getElementById('subtask-index').textContent =
      d.subtask_idx + '/' + d.total_subtasks;
    const typeEl = document.getElementById('subtask-type');
    typeEl.textContent = d.subtask_type || '-';
        typeEl.className = d.subtask_type || '';
    document.getElementById('subtask-prompt').textContent = d.subtask_prompt || '';
    document.getElementById('extra-info').textContent = d.extra_info || '';
    document.getElementById('status').textContent =
      'updated ' + new Date().toLocaleTimeString();

    const container = document.getElementById('cameras');
    const cams = d.cameras || [];
    // rebuild tiles only when camera count changes
    if (container.children.length !== cams.length) {
      container.innerHTML = '';
      cams.forEach(c => {
        const div = document.createElement('div');
        div.className = 'cam-tile';
        const img = document.createElement('img');
        img.id = 'img-' + c.name;
        const lbl = document.createElement('div');
        lbl.className = 'cam-label';
        lbl.textContent = c.name;
        div.appendChild(img);
        div.appendChild(lbl);
        container.appendChild(div);
      });
    }
    cams.forEach(c => {
      const img = document.getElementById('img-' + c.name);
      if (img) img.src = 'data:image/jpeg;base64,' + c.jpeg_b64;
    });
  } catch(e) {
    document.getElementById('status').textContent = 'connection error';
  }
  setTimeout(poll, POLL_MS);
}
poll();
</script>
</body>
</html>"""


# ── HTTP handler ────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """Serves the index page and a JSON state endpoint."""

    def __init__(self, visualizer: ReplayVisualizer, *args, **kwargs):
        self._vis = visualizer
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", _INDEX_HTML.encode())
        elif self.path == "/api/state":
            data = self._vis._get_state_json()  # noqa: SLF001
            self._respond(200, "application/json", data)
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence per-request logs


# ── Main class ──────────────────────────────────────────────────────────────


class ReplayVisualizer:
    """Web-based tiled camera views with subtask overlay.

    Parameters
    ----------
    environment:
        The ``ReplayEnvironment`` providing camera images.
    enabled:
        When ``False``, all methods become no-ops.
    port:
        HTTP server port.
    """

    def __init__(
        self,
        environment: ReplayEnvironment,
        *,
        enabled: bool = True,
        port: int = _DEFAULT_PORT,
        save_path: str = "",
        video_font_path: Path | None = None,
    ) -> None:
        self._env = environment
        self._enabled = enabled
        self._closed = False
        self._save_path = save_path.strip()
        self._cam_names: tuple[str, ...] = environment.camera_names
        self._total_steps: int = environment.num_steps
        self._writer: cv2.VideoWriter | None = None
        self._writer_frame_size: tuple[int, int] | None = None
        self._video_fps = float(environment.fps) if environment.fps > 0 else 25.0
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
                _FONT_ENV_VAR,
                exc,
            )

        # Shared state (protected by lock)
        self._lock = threading.Lock()
        self._step: int = 0
        self._subtask_idx: int = 0
        self._total_subtasks: int = 1
        self._subtask_type: str = ""
        self._subtask_prompt: str = ""
        self._extra_info: str = ""
        self._camera_jpegs: dict[str, bytes] = {}
        self._done: bool = False

        # Start server
        if not enabled:
            return
        handler = partial(_Handler, self)
        self._server = HTTPServer(("0.0.0.0", port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logging.info("Replay visualizer: http://0.0.0.0:%d", port)
        if self._save_path:
            logging.info("Replay visualizer recording enabled: %s", self._save_path)

    # -- public API -----------------------------------------------------------

    def set_subtask_context(
        self,
        subtask_idx: int,
        total_subtasks: int,
        subtask_type: str,
        subtask_prompt: str,
        *,
        extra_info: str = "",
    ) -> None:
        with self._lock:
            self._subtask_idx = subtask_idx
            self._total_subtasks = total_subtasks
            self._subtask_type = subtask_type
            self._subtask_prompt = subtask_prompt
        self._extra_info = extra_info

    def update(self, step: int, *, extra_info: str = "") -> bool:
        """Push a new frame to the web UI.  Always returns ``True``."""
        if self._closed:
            return True
        if not self._enabled and not self._save_path:
            return True

        clamped = min(step, self._total_steps - 1)
        jpegs: dict[str, bytes] = {}
        bgr_frames: dict[str, np.ndarray] = {}
        for cam_name in self._cam_names:
            try:
                img = self._env.get_image(cam_name, clamped)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr_frames[cam_name] = img
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpegs[cam_name] = buf.tobytes()
            except Exception:  # noqa: BLE001
                pass

        with self._lock:
            self._step = step
            self._extra_info = extra_info
            self._camera_jpegs = jpegs

        if self._save_path and bgr_frames:
            frame = self._compose_record_frame(clamped, bgr_frames, extra_info)
            self._write_video_frame(frame)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._enabled:
            with self._lock:
                self._done = True
        else:
            with self._lock:
                self._done = True
            self._server.shutdown()
            logging.info("Replay visualizer: server stopped.")
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logging.info("Replay visualizer video saved: %s", self._save_path)

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
                frame = cv2.resize(
                    frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA
                )
            normalized.append(frame)

        cols = min(3, len(normalized)) if len(normalized) > 1 else 1
        rows = (len(normalized) + cols - 1) // cols
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, frame in enumerate(normalized):
            r = idx // cols
            c = idx % cols
            y0 = r * tile_h
            x0 = c * tile_w
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame

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
            draw,
            (16, 22),
            title,
            self._font_title,
            padding=(8, 5),
            fg=(245, 245, 245, 255),
            bg=(96, 96, 96, 122),
            box_width=box_width,
        )
        draw_text_box(
            draw,
            (16, 52),
            subtask,
            self._font_body,
            padding=(8, 4),
            fg=(185, 228, 255, 255),
            bg=(96, 96, 96, 122),
            box_width=box_width,
        )
        draw_text_box(
            draw,
            (16, 80),
            prompt_line,
            self._font_body,
            padding=(8, 5),
            fg=(224, 224, 224, 255),
            bg=(96, 96, 96, 118),
            box_width=box_width,
        )
        if extra_info:
            draw_text_box(
                draw,
                (16, 108),
                info_line,
                self._font_body,
                padding=(8, 5),
                fg=(166, 210, 255, 255),
                bg=(96, 96, 96, 118),
                box_width=box_width,
            )
        return composite_overlay_on_bgr(canvas, overlay)

    def _write_video_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            directory = os.path.dirname(self._save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            self._writer = cv2.VideoWriter(
                self._save_path, fourcc, self._video_fps, (w, h)
            )
            self._writer_frame_size = (w, h)
            if not self._writer.isOpened():
                logging.error(
                    "Replay visualizer failed to open video writer: %s", self._save_path
                )
                self._writer.release()
                self._writer = None
                self._writer_frame_size = None
                return

        if self._writer_frame_size is None:
            return

        target_w, target_h = self._writer_frame_size
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        self._writer.write(frame)

    # -- internals (called from HTTP handler) ---------------------------------

    def _get_state_json(self) -> bytes:
        with self._lock:
            cameras = [
                {"name": name, "jpeg_b64": base64.b64encode(data).decode()}
                for name, data in self._camera_jpegs.items()
            ]
            state = {
                "step": self._step,
                "total_steps": self._total_steps,
                "subtask_idx": self._subtask_idx,
                "total_subtasks": self._total_subtasks,
                "subtask_type": self._subtask_type,
                "subtask_prompt": self._subtask_prompt,
                "extra_info": self._extra_info,
                "cameras": cameras,
                "done": self._done,
            }
        return json.dumps(state).encode()
