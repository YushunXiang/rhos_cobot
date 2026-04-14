"""Shared Pillow-based overlay rendering for feat-gui video producers.

This module provides font discovery, image-array conversion, and
drawing primitives built on Pillow's ``ImageDraw``, replacing the
OpenCV overlay paths in the feat-gui video producers. See
``docs/superpowers/specs/2026-04-14-pillow-overlay-migration-design.md``
for the full design.
"""
from __future__ import annotations


class FontUnavailableError(RuntimeError):
    """Raised when no usable TrueType font can be resolved."""
