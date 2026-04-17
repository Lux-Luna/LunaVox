"""Small reusable GUI widgets.

Each widget here is a thin CTkFrame subclass that knows how to render
itself and how to report its own state — no hidden globals, no
parent-class bookkeeping. Views import them as building blocks.
"""

from __future__ import annotations

from .conversion_dialog import ConversionDialog
from .param_slider import FieldSpec, ParamSliderGroup
from .stats_card import StatsCard
from .voice_picker import VoicePicker

__all__ = [
    "ConversionDialog",
    "FieldSpec",
    "ParamSliderGroup",
    "StatsCard",
    "VoicePicker",
]
