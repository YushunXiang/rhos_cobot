from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _install_cv2_stub() -> None:
    cv2_stub = types.SimpleNamespace()
    cv2_stub.COLOR_RGB2BGR = object()

    def cvt_color(frame: np.ndarray, code: object) -> np.ndarray:
        if code is not cv2_stub.COLOR_RGB2BGR:
            raise AssertionError(f"unexpected conversion code: {code!r}")
        return frame[..., ::-1].copy()

    cv2_stub.cvtColor = cvt_color
    sys.modules["cv2"] = cv2_stub


def _install_h5py_stub() -> None:
    sys.modules["h5py"] = types.SimpleNamespace(File=object)


def _load_module():
    _install_cv2_stub()
    _install_h5py_stub()
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_replay_camera_video.py"
    spec = importlib.util.spec_from_file_location("extract_replay_camera_video", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_frame_for_output_converts_rgb_to_bgr():
    module = _load_module()
    rgb_frame = np.array([[[255, 10, 1], [0, 20, 200]]], dtype=np.uint8)

    converted = module._prepare_frame_for_output(rgb_frame, input_color_space="rgb")

    expected = np.array([[[1, 10, 255], [200, 20, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(converted, expected)


def test_prepare_frame_for_output_keeps_bgr_frames():
    module = _load_module()
    bgr_frame = np.array([[[1, 10, 255], [200, 20, 0]]], dtype=np.uint8)

    converted = module._prepare_frame_for_output(bgr_frame, input_color_space="bgr")

    np.testing.assert_array_equal(converted, bgr_frame)
