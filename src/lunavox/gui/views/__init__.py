"""Top-level GUI views (one per sidebar entry)."""

from __future__ import annotations

from .library_view import LibraryView
from .settings_view import SettingsView
from .synth_view import SynthView

__all__ = ["LibraryView", "SettingsView", "SynthView"]
