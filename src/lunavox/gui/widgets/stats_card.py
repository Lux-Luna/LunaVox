"""Compact visualisation of a single :class:`SynthesisStats`.

Reads the dataclass directly — the old GUI converted stats back to a
"legacy metrics dict" before rendering, which drifted from the
authoritative field names every time C API changed. This version
uses the fields as-is.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from lunavox.runtime import SynthesisStats

from ..i18n import Translator
from ..theme import BG_CARD, CORNER_RADIUS, FONT_BODY, FONT_HEADING, SPACE_MD, SPACE_SM, TEXT_MUTED


class StatsCard(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, master: Any, translator: Translator) -> None:
        super().__init__(master, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        self._t = translator
        self._rows: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(self, text=self._t("stats.title"), font=FONT_HEADING).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM)
        )

        keys = [
            ("stats.rtf", "rtf"),
            ("stats.duration", "duration"),
            ("stats.total", "total"),
            ("stats.tokenize", "tokenize"),
            ("stats.encode", "encode"),
            ("stats.generate", "generate"),
            ("stats.decode", "decode"),
            ("stats.ram", "ram"),
        ]
        for i, (label_key, row_key) in enumerate(keys, start=1):
            ctk.CTkLabel(self, text=self._t(label_key), font=FONT_BODY, text_color=TEXT_MUTED).grid(
                row=i, column=0, sticky="w", padx=SPACE_MD, pady=2
            )
            value_label = ctk.CTkLabel(self, text="—", font=FONT_BODY)
            value_label.grid(row=i, column=1, sticky="e", padx=SPACE_MD, pady=2)
            self._rows[row_key] = value_label

        self.grid_columnconfigure(1, weight=1)

    def update_stats(self, stats: Optional[SynthesisStats]) -> None:
        if stats is None:
            for label in self._rows.values():
                label.configure(text="—")
            return
        self._rows["rtf"].configure(text=f"{stats.rtf:.3f}")
        self._rows["duration"].configure(text=f"{stats.audio_duration_ms / 1000:.2f} s")
        self._rows["total"].configure(text=f"{stats.t_total_ms} ms")
        self._rows["tokenize"].configure(text=f"{stats.t_tokenize_ms} ms")
        self._rows["encode"].configure(text=f"{stats.t_encode_ms} ms")
        self._rows["generate"].configure(text=f"{stats.t_generate_ms} ms")
        self._rows["decode"].configure(text=f"{stats.t_decode_ms} ms")
        self._rows["ram"].configure(text=f"{stats.rss_peak_bytes / (1024**3):.2f} GB")
