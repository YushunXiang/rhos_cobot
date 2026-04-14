"""Tests for rhos_cobot.pillow_overlay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


def test_module_exports_font_unavailable_error():
    from rhos_cobot import pillow_overlay as po

    assert issubclass(po.FontUnavailableError, RuntimeError)


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


def test_bgr_to_pil_color_swap():
    from rhos_cobot import pillow_overlay as po

    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
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
        draw,
        (10, 20),
        "x",
        _default_font(),
        bg=(50, 50, 50),
        box_width=200,
    )
    assert bbox[2] - bbox[0] == 200


def test_draw_text_box_with_bg_paints_background():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (100, 40), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    po.draw_text_box(
        draw,
        (10, 15),
        "Y",
        _default_font(),
        fg=(255, 255, 255),
        bg=(40, 80, 160),
    )
    arr = np.asarray(img)
    blue_pixels = np.all(arr == (40, 80, 160), axis=-1)
    assert blue_pixels.any()


def test_max_text_width_returns_widest():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (300, 80))
    draw = ImageDraw.Draw(img)
    wide = po.max_text_width(draw, _default_font(), ["a", "longer string", "b"], padding_x=4)
    short = po.max_text_width(draw, _default_font(), ["a"], padding_x=4)
    assert wide > short


def test_max_text_width_ignores_empty_strings():
    from rhos_cobot import pillow_overlay as po

    img = Image.new("RGB", (100, 40))
    draw = ImageDraw.Draw(img)
    assert po.max_text_width(draw, _default_font(), ["", ""], padding_x=4) == 0
