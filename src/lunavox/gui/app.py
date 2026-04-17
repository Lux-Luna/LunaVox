"""Top-level application: window chrome, sidebar navigation, view switching.

The app is a single :class:`customtkinter.CTk` window with a fixed
sidebar on the left and one of three views rendered on the right.
Switching views recreates the target view so data-dependent fields
(library listings, settings dropdowns) stay fresh.
"""

from __future__ import annotations

from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from lunavox.cli._config import ResolvedConfig, load_config

from .engine_session import EngineSession
from .i18n import Translator
from .theme import (
    BG_SIDEBAR,
    BG_SURFACE,
    FONT_BODY,
    FONT_TITLE,
    PRIMARY,
    PRIMARY_HOVER,
    SIDEBAR_WIDTH,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    TEXT_MUTED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from .views import LibraryView, SettingsView, SynthView


class LunaVoxApp(ctk.CTk):  # pyright: ignore[reportUntypedBaseClass]
    """LunaVox desktop GUI.

    Pass a :class:`ResolvedConfig` to reuse the CLI's config loader;
    passing ``None`` loads the default profile so the window can be
    launched directly (``python -m lunavox.gui``).
    """

    def __init__(self, config: Optional[ResolvedConfig] = None) -> None:
        super().__init__()
        self._config = config or load_config()
        self._t = Translator(lang="en")
        self._active_view: Optional[Any] = None
        self._nav_buttons: dict[str, Any] = {}
        # Owns the optional persistent Engine. Lives on the app, not on
        # SynthView, because _show_view destroys views on every tab
        # switch — pinning the engine there would re-cold-load it.
        self._engine_session = EngineSession()

        self.title("LunaVox")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(1080, 640)
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=BG_SURFACE)

        self._build_sidebar()
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.grid(row=0, column=1, sticky="nsew", padx=SPACE_LG, pady=SPACE_LG)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Free the persistent engine (if any) before the Tk window
        # tears down, so the C library finalises cleanly instead of
        # surviving past the interpreter's mainloop exit.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._show_view("synthesize")

    # --- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, width=SIDEBAR_WIDTH, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(sidebar, text=self._t("app.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, 0)
        )
        ctk.CTkLabel(
            sidebar,
            text=self._t("app.subtitle"),
            font=FONT_BODY,
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG))

        entries = [
            ("synthesize", "nav.synthesize"),
            ("library", "nav.library"),
            ("settings", "nav.settings"),
        ]
        for row, (view_id, label_key) in enumerate(entries, start=2):
            btn = ctk.CTkButton(
                sidebar,
                text=self._t(label_key),
                anchor="w",
                fg_color="transparent",
                text_color=("#1A202C", "#E2E8F0"),
                hover_color=PRIMARY_HOVER,
                corner_radius=8,
                height=40,
                command=lambda v=view_id: self._show_view(v),
            )
            btn.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=2)
            self._nav_buttons[view_id] = btn

        # Bottom-aligned footer.
        ctk.CTkLabel(
            sidebar,
            text="v2.2.2",
            font=FONT_BODY,
            text_color=TEXT_MUTED,
        ).grid(row=11, column=0, sticky="sw", padx=SPACE_LG, pady=SPACE_SM)

    def _show_view(self, view_id: str) -> None:
        # Drop any existing view so the replacement starts from a
        # clean grid — recreating is cheap and avoids stale references
        # to config fields the user may have just edited.
        for child in self._content.winfo_children():
            child.destroy()

        for vid, btn in self._nav_buttons.items():
            btn.configure(fg_color=PRIMARY if vid == view_id else "transparent")

        if view_id == "synthesize":
            view = SynthView(
                self._content, self._config, self._t, engine_session=self._engine_session
            )
        elif view_id == "library":
            view = LibraryView(self._content, self._config, self._t)
        elif view_id == "settings":
            view = SettingsView(
                self._content,
                self._config,
                self._t,
                on_lang_changed=self._set_language,
                engine_session=self._engine_session,
            )
        else:
            return
        view.pack(fill="both", expand=True)
        self._active_view = view

    # --- language toggle ----------------------------------------------

    def _set_language(self, lang: str) -> None:
        self._t.set_lang(lang)
        # Rebuilding the sidebar + current view picks up every
        # translated string without a per-widget refresh pass.
        for child in self.winfo_children():
            child.destroy()
        self._nav_buttons = {}
        self._build_sidebar()
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.grid(row=0, column=1, sticky="nsew", padx=SPACE_LG, pady=SPACE_LG)
        self._show_view("settings")

    # --- lifecycle -----------------------------------------------------

    def _on_close(self) -> None:
        self._engine_session.unload()
        self.destroy()

    # --- entrypoint ----------------------------------------------------

    def run(self) -> None:
        self.mainloop()


def main() -> None:
    LunaVoxApp().run()


if __name__ == "__main__":
    main()
