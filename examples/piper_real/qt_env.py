# -- coding: UTF-8
"""When cv2 and PySide6 are both in the venv, OpenCV can point Qt at cv2/qt/plugins and break the xcb plugin.

Call `fix_qt_for_matplotlib()` before importing any module that loads `cv2` or `matplotlib` Qt backends.

On Ubuntu/Debian, Qt 6.5+ on X11 also needs: `sudo apt install libxcb-cursor0` (xcb-cursor).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_DONE = False


def fix_qt_for_matplotlib() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True

    qpp = os.environ.get("QT_PLUGIN_PATH", "")
    if qpp and ("cv2" in qpp or "/cv2/" in qpp):
        del os.environ["QT_PLUGIN_PATH"]
    # Pin platform plugins to PySide6, not opencv.
    try:
        spec = importlib.util.find_spec("PySide6")
        if spec and spec.origin:
            root = Path(spec.origin).resolve().parent
            plug = root / "Qt" / "plugins"
            if plug.is_dir():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plug)
    except Exception:  # noqa: BLE001
        pass
