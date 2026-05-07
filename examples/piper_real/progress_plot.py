# -- coding: UTF-8
"""Live dual-curve plot for task_progress and subtask_progress during policy inference."""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import warnings
from collections import deque
from typing import Deque, List, Optional

try:
    from examples.piper_real import qt_env as _qt_env
except ModuleNotFoundError:
    import qt_env as _qt_env

_qt_env.fix_qt_for_matplotlib()

import numpy as np

# Set backend *before* most pyplot usage. `matplotlib.use("TkAgg")` can succeed even when
# tkinter is broken; failure often happens on first `figure()`. We smoke-test each candidate.
import matplotlib


def _ensure_display_env() -> None:
    """IDE/ssh-less terminals often have no DISPLAY even on a desktop; Wayland may omit DISPLAY."""
    if os.name == "nt":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    # Remote SSH without X11 forwarding: do not guess :0 (would be wrong / hang on some hosts).
    if os.environ.get("SSH_CONNECTION"):
        return
    # Local session (typical Cursor/VSCode integrated terminal on Linux): default X11 socket.
    os.environ.setdefault("DISPLAY", ":0")


def _smoke_figure() -> None:
    import matplotlib.pyplot as plt

    plt.close("all")
    fig = plt.figure()
    plt.close(fig)


def _install_matplotlib_backend() -> str:
    """Prefer Qt* before Tk* (many Linux images lack a working _tkinter / tkinter)."""
    _ensure_display_env()
    env_raw = (os.environ.get("MPLBACKEND") or "").strip()
    env = env_raw.lower()
    has_display = (
        bool(os.environ.get("DISPLAY", ""))
        or bool(os.environ.get("WAYLAND_DISPLAY", ""))
        or os.name == "nt"
    )
    headless = env == "agg" and not has_display and os.name != "nt"

    if headless:
        candidates: List[str] = ["Agg"]
    elif not env:
        # QtAgg first: picks PySide6/PyQt6 when available; Qt5Agg needs PyQt5.
        candidates = ["QtAgg", "Qt5Agg", "TkAgg", "WebAgg", "macosx", "Agg"]
    elif env == "agg" and has_display:
        candidates = ["QtAgg", "Qt5Agg", "TkAgg", "WebAgg", "macosx", "Agg"]
    else:
        candidates = [env_raw, "QtAgg", "Qt5Agg", "TkAgg", "WebAgg", "macosx", "Agg"]

    failures: List[tuple[str, str]] = []
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            if "matplotlib.pyplot" in sys.modules:
                importlib.reload(sys.modules["matplotlib.pyplot"])
            _smoke_figure()
            return matplotlib.get_backend()
        except Exception as e:  # noqa: BLE001
            last_err = e
            failures.append((name, f"{type(e).__name__}: {e}"))
    if failures:
        print("[plot_progress] Backend smoke-tests failed:", flush=True)
        for n, msg in failures[-5:]:
            print(f"  - {n}: {msg}", flush=True)
        if last_err is not None:
            print(f"[plot_progress] Last error: {last_err!r}", flush=True)
    matplotlib.use("Agg", force=True)
    if "matplotlib.pyplot" in sys.modules:
        importlib.reload(sys.modules["matplotlib.pyplot"])
    _smoke_figure()
    return "Agg"


_install_matplotlib_backend()
import matplotlib.pyplot as plt
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

# True FigureCanvasAgg (not TkAgg) cannot display a window; use file fallback.
try:
    from matplotlib.backends import backend_agg
except Exception:  # noqa: BLE001
    backend_agg = None


def _first_scalar(x: object) -> Optional[float]:
    if x is None:
        return None
    a = np.asarray(x).ravel()
    if a.size == 0:
        return None
    v = float(a.flat[0])
    if np.isnan(v) or np.isinf(v):
        return None
    return v


class DualProgressPlotSubscriber(_subscriber.Subscriber):
    """Non-blocking matplotlib window: two lines (task + subtask), rolling window in x."""

    def __init__(self, window: int = 500, update_every: int = 1) -> None:
        self._window = max(2, int(window))
        self._update_every = max(1, int(update_every))
        self._frame = 0
        self._step_idx = 0
        self._buf_max = max(2000, self._window * 3)
        self._x: Deque[int] = deque(maxlen=self._buf_max)
        self._y_task: Deque[float] = deque(maxlen=self._buf_max)
        self._y_sub: Deque[float] = deque(maxlen=self._buf_max)
        self._hold_task: Optional[float] = None
        self._hold_sub: Optional[float] = None
        self._file_path: Optional[str] = None

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(8.5, 4.0), num="Progress (live)")
        (self._line_task,) = self._ax.plot(
            [],
            [],
            color="C0",
            linewidth=1.5,
            label="task_progress",
        )
        (self._line_sub,) = self._ax.plot(
            [],
            [],
            color="C1",
            linewidth=1.5,
            label="subtask_progress",
        )
        self._ax.set_ylim(-0.05, 1.05)
        self._ax.set_xlabel("Control step")
        self._ax.set_ylabel("progress")
        self._ax.set_title("Task / subtask progress (live)")
        self._ax.grid(True, alpha=0.35)
        self._ax.legend(loc="lower right", fontsize=9)
        self._status = self._ax.text(
            0.01,
            0.99,
            "",
            transform=self._ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        self._fig.tight_layout()

        _is_agg = (
            backend_agg is not None and isinstance(self._fig.canvas, backend_agg.FigureCanvasAgg)
        )
        if _is_agg:
            self._file_path = os.path.join(tempfile.gettempdir(), "rhos_piper_progress.png")
            _disp = os.environ.get("DISPLAY", "")
            _wl = os.environ.get("WAYLAND_DISPLAY", "")
            print(
                "[plot_progress] Matplotlib is using the non-interactive Agg backend; "
                f"curves are written to {self._file_path} (overwritten each update).\n"
                f"  Environment: DISPLAY={_disp!r} WAYLAND_DISPLAY={_wl!r} MPLBACKEND="
                f"{os.environ.get('MPLBACKEND', '')!r}\n"
                "  For a live window: (1) In the **same venv**: `pip install pyside6` (or pyqt5);\n"
                "  (2) If the terminal has no DISPLAY (common in IDE), run in a normal "
                "terminal or: `export DISPLAY=:0` (or your session’s value from `echo $DISPLAY` in a "
                "desktop terminal);\n"
                "  (3) Remote: `ssh -X` and ensure DISPLAY is set; optional `export MPLBACKEND=QtAgg`.",
                flush=True,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._fig.show()

    @override
    def on_episode_start(self) -> None:
        self._frame = 0
        self._step_idx = 0
        self._x = deque(maxlen=self._buf_max)
        self._y_task = deque(maxlen=self._buf_max)
        self._y_sub = deque(maxlen=self._buf_max)
        self._hold_task = None
        self._hold_sub = None
        self._line_task.set_data([], [])
        self._line_sub.set_data([], [])
        self._status.set_text("Waiting for first progress values…")
        self._ax.set_xlim(0, self._window)
        self._redraw(silent=True)

    @override
    def on_episode_end(self) -> None:
        self._redraw(silent=True)

    @override
    def on_step(self, observation: dict, action: dict) -> None:  # noqa: ARG002
        self._step_idx += 1
        t_new = _first_scalar(action.get("task_progress")) if isinstance(action, dict) else None
        s_new = _first_scalar(action.get("subtask_progress")) if isinstance(action, dict) else None
        if t_new is None and isinstance(action, dict) and "progress" in action:
            t_new = _first_scalar(action.get("progress"))
        if t_new is not None:
            self._hold_task = t_new
        if s_new is not None:
            self._hold_sub = s_new
        if self._hold_task is None or self._hold_sub is None:
            return

        u, v = self._hold_task, self._hold_sub
        self._x.append(self._step_idx)
        self._y_task.append(u)
        self._y_sub.append(v)

        self._frame += 1
        if self._frame % self._update_every != 0:
            return

        x_list: List[int] = list(self._x)
        self._line_task.set_data(x_list, list(self._y_task))
        self._line_sub.set_data(x_list, list(self._y_sub))
        if x_list:
            lo = max(0, x_list[-1] - self._window)
            self._ax.set_xlim(lo, max(float(lo) + 1, float(x_list[-1]) + 1.0))
        self._status.set_text(
            f"task={u:.3f}  subtask={v:.3f}  (step {self._step_idx})",
        )
        self._redraw(silent=False)

    def _redraw(self, *, silent: bool) -> None:
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            if not silent:
                if self._file_path:
                    self._fig.savefig(self._file_path, dpi=120, bbox_inches="tight")
                else:
                    plt.pause(0.001)
        except Exception:
            pass
