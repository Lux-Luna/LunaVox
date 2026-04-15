"""The library view — installed models and reference voices.

Pure read-only surface: both lists come from scanning directories on
disk. Editing / pulling / deleting is a CLI concern (``lunavox model
pull``). Keeping the GUI thin here means there is only one place that
knows how to download models.
"""

from __future__ import annotations

from typing import Any

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from lunavox.cli._config import ResolvedConfig
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


class LibraryView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, master: Any, config: ResolvedConfig, translator: Translator) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self._t("lib.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_MD)
        )

        # --- Models section ---
        models_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        models_card.grid(row=1, column=0, sticky="ew", pady=(0, SPACE_MD))
        models_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(models_card, text=self._t("lib.models_section"), font=FONT_HEADING).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        models_dir = self._config.project_root / "models"
        specs = all_models()
        rows_added = 0
        for i, spec in enumerate(specs, start=1):
            local = models_dir / spec.name
            installed = local.exists() and any(local.iterdir())
            badge = "✓" if installed else "—"
            ctk.CTkLabel(
                models_card,
                text=f"{badge}  {spec.display_name}",
                font=FONT_BODY,
            ).grid(row=i, column=0, sticky="w", padx=SPACE_LG, pady=2)
            ctk.CTkLabel(
                models_card,
                text=spec.repo_id,
                font=FONT_BODY,
                text_color=TEXT_MUTED,
            ).grid(row=i, column=1, sticky="e", padx=SPACE_LG, pady=2)
            rows_added = i

        if rows_added == 0:
            ctk.CTkLabel(models_card, text=self._t("lib.no_models"), text_color=TEXT_MUTED).grid(
                row=1, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_MD
            )

        ctk.CTkLabel(models_card, text="").grid(row=rows_added + 1, column=0, pady=(0, SPACE_SM))

        # --- References section ---
        refs_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        refs_card.grid(row=2, column=0, sticky="ew")
        refs_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(refs_card, text=self._t("lib.references_section"), font=FONT_HEADING).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        ref_dir = self._config.project_root / "ref"
        if ref_dir.exists():
            files = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in {".wav", ".json"})
        else:
            files = []

        if not files:
            ctk.CTkLabel(refs_card, text=self._t("lib.no_references"), text_color=TEXT_MUTED).grid(
                row=1, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG)
            )
        else:
            for i, p in enumerate(files, start=1):
                ctk.CTkLabel(refs_card, text=p.name, font=FONT_BODY).grid(
                    row=i, column=0, sticky="w", padx=SPACE_LG, pady=2
                )
            ctk.CTkLabel(refs_card, text="").grid(row=len(files) + 1, column=0, pady=(0, SPACE_SM))
