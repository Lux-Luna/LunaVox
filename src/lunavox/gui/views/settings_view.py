"""Settings — threads, language, and a readout of the active native runtimes.

The CLI's TOML profiles are intentionally absent from this view: the
GUI only edits session-scoped knobs and points at the Library view for
swapping the llama.cpp / onnxruntime backend on disk.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from lunavox.cli._config import ResolvedConfig

from ..engine_session import EngineSession
from ..i18n import Translator
from ..lib_info import read_lib_metadata
from ..theme import (
    BG_CARD,
    CORNER_RADIUS,
    FONT_BODY,
    FONT_HEADING,
    FONT_TITLE,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    TEXT_MUTED,
)


class SettingsView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(
        self,
        master: Any,
        config: ResolvedConfig,
        translator: Translator,
        on_lang_changed: Optional[Callable[[str], None]] = None,
        engine_session: Optional[EngineSession] = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._on_lang_changed = on_lang_changed
        # Optional so the view can still render in isolation (tests,
        # the `python -m lunavox.gui.views.settings_view` debug path)
        # but production always wires a real session in from the app.
        self._session = engine_session
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self._t("settings.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_MD)
        )

        # --- General (threads + apply) ---
        general = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        general.grid(row=1, column=0, sticky="ew")
        general.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(general, text=self._t("settings.threads_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )
        self._threads_var = ctk.IntVar(value=self._config.n_threads)
        ctk.CTkEntry(general, textvariable=self._threads_var, width=80).grid(
            row=0, column=1, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        self._status_label = ctk.CTkLabel(general, text="", font=FONT_BODY, text_color=TEXT_MUTED)
        self._status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=SPACE_LG)

        ctk.CTkButton(general, text=self._t("settings.apply"), command=self._apply).grid(
            row=2, column=0, columnspan=2, sticky="e", padx=SPACE_LG, pady=(SPACE_SM, SPACE_MD)
        )

        # --- Active runtimes (read-only readout from lib/metadata.json) ---
        runtimes = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        runtimes.grid(row=2, column=0, sticky="ew", pady=(SPACE_MD, 0))
        runtimes.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(runtimes, text=self._t("settings.runtimes_label"), font=FONT_HEADING).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        meta = read_lib_metadata(self._config.project_root)
        self._add_runtime_row(runtimes, 1, self._t("settings.runtime_llama"), meta.get("llama"))
        self._add_runtime_row(runtimes, 2, self._t("settings.runtime_onnx"), meta.get("onnx"))
        ctk.CTkLabel(runtimes, text="").grid(row=3, column=0, columnspan=2, pady=(0, SPACE_SM))

        # --- Language toggle ---
        lang_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        lang_card.grid(row=3, column=0, sticky="ew", pady=(SPACE_MD, 0))
        ctk.CTkLabel(lang_card, text="Language / 语言", font=FONT_HEADING).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )
        ctk.CTkButton(
            lang_card,
            text=self._t("lang.toggle"),
            command=self._toggle_language,
        ).grid(row=1, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG))

    def _add_runtime_row(self, parent: Any, row: int, label: str, info: Any) -> None:
        ctk.CTkLabel(parent, text=label, font=FONT_BODY).grid(
            row=row, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        if info is None:
            value = self._t("settings.runtime_unknown")
        else:
            value = f"{info.label} · {info.version}"
        ctk.CTkLabel(parent, text=value, font=FONT_BODY, text_color=TEXT_MUTED).grid(
            row=row, column=1, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )

    def _apply(self) -> None:
        old_threads = self._config.n_threads
        with contextlib.suppress(TypeError, ValueError):
            self._config.n_threads = int(self._threads_var.get())
        # n_threads is a construction-time arg on Engine, so a cached
        # pre-loaded engine doesn't pick it up. Drop the cache on
        # change — when the user flips back to Synth, the Pre-load
        # checkbox reads session state and shows itself unchecked.
        if self._session is not None and self._config.n_threads != old_threads:
            self._session.unload()
        self._status_label.configure(text=self._t("settings.applied"))

    def _toggle_language(self) -> None:
        new_lang = "zh" if self._t.lang == "en" else "en"
        if self._on_lang_changed is not None:
            self._on_lang_changed(new_lang)
