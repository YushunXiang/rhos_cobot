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
import threading
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

import cv2
import numpy as np

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
  body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; }
  #header {
    padding: 12px 20px; background: #16213e;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #0f3460;
  }
  #header h1 { font-size: 16px; font-weight: 600; color: #e94560; }
  #progress { font-size: 13px; color: #a0a0b0; }
  #subtask-bar {
    padding: 10px 20px; background: #0f3460;
    border-bottom: 1px solid #1a1a4e;
  }
  #subtask-bar .label { font-size: 11px; color: #7a7a9a; text-transform: uppercase; letter-spacing: 0.5px; }
  #subtask-bar .type { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin: 0 6px; }
  #subtask-bar .type.navigate { background: #1b998b; color: #fff; }
  #subtask-bar .type.manipulate { background: #e94560; color: #fff; }
  #subtask-bar .type.policy { background: #6c63ff; color: #fff; }
  #subtask-prompt { font-size: 13px; margin-top: 4px; color: #ccc; }
  #extra-info { font-size: 12px; margin-top: 3px; color: #8ab4f8; }
  #cameras {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 8px; padding: 12px;
  }
  .cam-tile { position: relative; }
  .cam-tile img { display: block; max-height: 70vh; width: auto; border-radius: 4px; }
  .cam-label {
    position: absolute; top: 6px; left: 8px;
    background: rgba(0,0,0,0.6); color: #fff; font-size: 11px;
    padding: 2px 6px; border-radius: 3px;
  }
  #status { text-align: center; padding: 8px; font-size: 12px; color: #555; }
</style>
</head>
<body>
  <div id="header">
    <h1>Replay Visualizer</h1>
    <span id="progress">Step -/-</span>
  </div>
  <div id="subtask-bar">
    <span class="label">Subtask</span>
    <span id="subtask-index">-/-</span>
    <span id="subtask-type" class="type">-</span>
    <div id="subtask-prompt">-</div>
    <div id="extra-info"></div>
  </div>
  <div id="cameras"></div>
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
    typeEl.className = 'type ' + (d.subtask_type || '');
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
    ) -> None:
        self._env = environment
        self._enabled = enabled
        self._closed = False
        self._cam_names: tuple[str, ...] = environment.camera_names
        self._total_steps: int = environment.num_steps

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

    # -- public API -----------------------------------------------------------

    def set_subtask_context(
        self,
        subtask_idx: int,
        total_subtasks: int,
        subtask_type: str,
        subtask_prompt: str,
    ) -> None:
        with self._lock:
            self._subtask_idx = subtask_idx
            self._total_subtasks = total_subtasks
            self._subtask_type = subtask_type
            self._subtask_prompt = subtask_prompt

    def update(self, step: int, *, extra_info: str = "") -> bool:
        """Push a new frame to the web UI.  Always returns ``True``."""
        if not self._enabled or self._closed:
            return True

        clamped = min(step, self._total_steps - 1)
        jpegs: dict[str, bytes] = {}
        for cam_name in self._cam_names:
            try:
                img = self._env.get_image(cam_name, clamped)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpegs[cam_name] = buf.tobytes()
            except Exception:  # noqa: BLE001
                pass

        with self._lock:
            self._step = step
            self._extra_info = extra_info
            self._camera_jpegs = jpegs
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._enabled:
            return
        with self._lock:
            self._done = True
        self._server.shutdown()
        logging.info("Replay visualizer: server stopped.")

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
