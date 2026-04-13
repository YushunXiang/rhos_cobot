from __future__ import annotations

import importlib
import os
import sys


def test_visualize_test_module_defers_filesystem_access_until_main(monkeypatch):
    sys.modules.pop("scripts.post_collect.visualize_test", None)

    def _unexpected_listdir(_path: str):
        raise AssertionError("os.listdir should not run during module import")

    monkeypatch.setattr(os, "listdir", _unexpected_listdir)

    module = importlib.import_module("scripts.post_collect.visualize_test")

    assert hasattr(module, "main")
