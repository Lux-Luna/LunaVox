"""Build view — kick off the native C++ engine compile from the GUI.

The actual build pipeline lives in :mod:`lunavox.build.main` and is
already wired to print stage banners + verbose output to stdout. We
shell out to ``python -m lunavox.build.main`` so the existing Rich
formatting flows straight into the log dialog without any duplication.
"""

from __future__ import annotations

import sys
from typing import Any

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from lunavox.cli._config import ResolvedConfig

from ..i18n import Translator
from ..theme import (
    BG_CARD,
    CORNER_RADIUS,
    FONT_BODY,
    FONT_HEADING,
    FONT_TITLE,
    PRIMARY,
    PRIMARY_HOVER,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    TEXT_MUTED,
)
from ..widgets import ConversionDialog

_TOOLCHAINS = ["auto", "msvc", "mingw", "clang", "gcc"]


class BuildView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, master: Any, config: ResolvedConfig, translator: Translator) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._build_btn: Any = None
        self._status: Any = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self._t("build.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )
        ctk.CTkLabel(
            self,
            text=self._t("build.subtitle"),
            font=FONT_BODY,
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(0, SPACE_MD))

        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        card.grid(row=2, column=0, sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        # Clean checkbox
        self._clean_var = ctk.BooleanVar(value=False)
        ctk.CTkLabel(card, text=self._t("build.clean_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )
        ctk.CTkCheckBox(card, text="", variable=self._clean_var).grid(
            row=0, column=1, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        # Parallel jobs
        ctk.CTkLabel(card, text=self._t("build.jobs_label"), font=FONT_BODY).grid(
            row=1, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        self._jobs_var = ctk.IntVar(value=max(1, self._config.n_threads))
        ctk.CTkEntry(card, textvariable=self._jobs_var, width=80).grid(
            row=1, column=1, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )

        # Toolchain
        ctk.CTkLabel(card, text=self._t("build.toolchain_label"), font=FONT_BODY).grid(
            row=2, column=0, sticky="w", padx=SPACE_LG, pady=SPACE_SM
        )
        self._toolchain_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(
            card, values=_TOOLCHAINS, variable=self._toolchain_var, width=160
        ).grid(row=2, column=1, sticky="w", padx=SPACE_LG, pady=SPACE_SM)

        # Run button + idle status
        self._status = ctk.CTkLabel(
            card,
            text=self._t("build.idle"),
            font=FONT_BODY,
            text_color=TEXT_MUTED,
        )
        self._status.grid(
            row=3, column=0, columnspan=2, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM)
        )

        self._build_btn = ctk.CTkButton(
            card,
            text=self._t("build.run_btn"),
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            font=FONT_HEADING,
            height=40,
            command=self._on_run,
        )
        self._build_btn.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_LG)
        )

    def _on_run(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "lunavox.build.main",
            "--project-root",
            str(self._config.project_root),
            "--j",
            str(max(1, int(self._jobs_var.get() or 1))),
            "--toolchain",
            str(self._toolchain_var.get() or "auto"),
            "--verbose",
        ]
        if self._clean_var.get():
            cmd.append("--clean")

        self._build_btn.configure(state="disabled")
        self._status.configure(text=self._t("build.in_progress"))

        def on_done(success: bool) -> None:
            self._build_btn.configure(state="normal")
            self._status.configure(
                text=self._t("lib.task_done") if success else self._t("lib.task_failed").format(err="")
            )

        ConversionDialog(
            self,
            translator=self._t,
            title=self._t("build.running_title"),
            cmd=cmd,
            on_done=on_done,
        )
