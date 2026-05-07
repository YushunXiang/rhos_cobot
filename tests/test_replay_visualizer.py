"""Tests for examples.piper_real.replay_visualizer overlay behavior."""
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
        return frame

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


def _fake_env():
    env = types.SimpleNamespace()
    env.camera_names = ("cam_high",)
    env.num_steps = 5
    env.fps = 25.0

    def get_image(name: str, step: int) -> np.ndarray:
        return np.full((60, 80, 3), 120, dtype=np.uint8)

    env.get_image = get_image
    return env


def _load_visualizer_module(monkeypatch):
    _install_cv2_stub(monkeypatch)
    from examples.piper_real import replay_visualizer as rv

    return rv


def test_constructor_resolves_font_when_available(monkeypatch):
    rv = _load_visualizer_module(monkeypatch)
    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not dejavu.is_file():
        pytest.skip("no system font available")

    vis = rv.ReplayVisualizer(
        _fake_env(),
        enabled=False,
        save_path="",
        video_font_path=dejavu,
    )
    assert vis._font_title is not None
    assert vis._font_body is not None
    vis.close()


def test_constructor_degrades_when_font_missing(monkeypatch, caplog, tmp_path):
    """Font-missing must emit a loud warning and disable HUD."""
    rv = _load_visualizer_module(monkeypatch)
    from rhos_cobot import pillow_overlay as po

    monkeypatch.delenv(po._FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(po, "_CJK_FONT_CANDIDATES", (str(tmp_path / "absent.ttc"),))
    monkeypatch.setattr(po, "_LATIN_FONT_CANDIDATES", (str(tmp_path / "absent.ttf"),))

    with caplog.at_level("WARNING"):
        vis = rv.ReplayVisualizer(_fake_env(), enabled=False, save_path="")

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

    vis = rv.ReplayVisualizer(_fake_env(), enabled=False, save_path="")
    frame_bgr = np.full((60, 80, 3), 120, dtype=np.uint8)
    canvas = vis._compose_record_frame(0, {"cam_high": frame_bgr}, extra_info="")

    assert canvas.shape == frame_bgr.shape
    assert np.array_equal(canvas, frame_bgr)
    vis.close()
