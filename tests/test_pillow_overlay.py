"""Tests for rhos_cobot.pillow_overlay."""
from __future__ import annotations

import pytest


def test_module_exports_font_unavailable_error():
    from rhos_cobot import pillow_overlay as po

    assert issubclass(po.FontUnavailableError, RuntimeError)
