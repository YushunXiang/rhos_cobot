"""Tests for rhos_cobot.pillow_overlay."""
from __future__ import annotations

from pathlib import Path

import pytest


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
