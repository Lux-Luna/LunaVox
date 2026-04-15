"""Read-only settings summary + language toggle.

Editing the config file is the CLI's job (or you edit
``~/.lunavox/config.toml`` directly). The GUI just shows what's
currently active so the user knows which profile is driving
synthesis. Applying changes here mutates the in-memory
:class:`ResolvedConfig` for the running session only — persistent
edits go through the TOML file on disk.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Optional

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from lunavox.cli._config import DEFAULT_CONFIG_PATH, ResolvedConfig
from lunavox.model import all_models

from ..i18n import Translator
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
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._on_lang_changed = on_lang_changed
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self._t("settings.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_MD)
        )

        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        card.grid(row=1, column=0, sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        row = 0
        row = self._add_row(
            card, row, self._t("settings.profile_label"), self._config.profile_name or "default"
        )

        # Model dropdown — mutates ResolvedConfig in place on apply.
        model_keys_list = [m.name for m in all_models()]
        ctk.CTkLabel(card, text=self._t("settings.model_label"), font=FONT_BODY).grid(
            row=row, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        self._model_var = ctk.StringVar(value=self._config.model)
        ctk.CTkOptionMenu(card, values=model_keys_list, variable=self._model_var).grid(
            row=row, column=1, sticky="ew", padx=SPACE_LG, pady=SPACE_SM
        )
        row += 1

        ctk.CTkLabel(card, text=self._t("settings.backend_label"), font=FONT_BODY).grid(
            row=row, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        self._backend_var = ctk.StringVar(value=self._config.backend)
        ctk.CTkOptionMenu(
            card, values=["auto", "cuda", "vulkan", "vulkan+dml", "cpu"], variable=self._backend_var
        ).grid(row=row, column=1, sticky="ew", padx=SPACE_LG, pady=SPACE_SM)
        row += 1

        ctk.CTkLabel(card, text=self._t("settings.threads_label"), font=FONT_BODY).grid(
            row=row, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        self._threads_var = ctk.IntVar(value=self._config.n_threads)
        ctk.CTkEntry(card, textvariable=self._threads_var, width=80).grid(
            row=row, column=1, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        row += 1

        row = self._add_row(
            card, row, self._t("settings.config_path_label"), str(DEFAULT_CONFIG_PATH)
        )

        # Apply + status line.
        self._status_label = ctk.CTkLabel(card, text="", font=FONT_BODY, text_color=TEXT_MUTED)
        self._status_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=SPACE_LG)
        row += 1

        ctk.CTkButton(card, text=self._t("settings.apply"), command=self._apply).grid(
            row=row, column=0, columnspan=2, sticky="e", padx=SPACE_LG, pady=SPACE_MD
        )

        # Language toggle lives in a separate card so it always shows
        # even if the main card is collapsed for smaller windows.
        lang_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        lang_card.grid(row=2, column=0, sticky="ew", pady=(SPACE_MD, 0))
        ctk.CTkLabel(lang_card, text="Language / 语言", font=FONT_HEADING).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )
        ctk.CTkButton(
            lang_card,
            text=self._t("lang.toggle"),
            command=self._toggle_language,
        ).grid(row=1, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG))

    def _add_row(self, parent: Any, row: int, label: str, value: str) -> int:
        ctk.CTkLabel(parent, text=label, font=FONT_BODY).grid(
            row=row, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        ctk.CTkLabel(parent, text=value, font=FONT_BODY, text_color=TEXT_MUTED).grid(
            row=row, column=1, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        return row + 1

    def _apply(self) -> None:
        self._config.model = self._model_var.get()
        self._config.backend = self._backend_var.get()
        # A bad entry shouldn't block the rest of the apply — keep the
        # old n_threads value and let the user correct it visually.
        with contextlib.suppress(TypeError, ValueError):
            self._config.n_threads = int(self._threads_var.get())
        self._status_label.configure(text=self._t("settings.applied"))

    def _toggle_language(self) -> None:
        new_lang = "zh" if self._t.lang == "en" else "en"
        if self._on_lang_changed is not None:
            self._on_lang_changed(new_lang)
