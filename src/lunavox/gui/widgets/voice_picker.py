"""A single widget that covers all four :class:`Voice` modes.

Before the refactor the GUI had four separate forms with partially
overlapping fields. :class:`VoicePicker` merges them: pick a mode
from a segmented control and the relevant inputs (reference path /
speaker id / instruction) become visible. Exactly one
:class:`lunavox.runtime.Voice` can be built at any moment.
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from lunavox.runtime import Voice

from ..i18n import Translator
from ..theme import FONT_BODY, SPACE_MD, SPACE_SM

_MODES = ("base", "clone", "custom", "design")


class VoicePicker(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    """Segmented voice-mode selector + the inputs each mode needs."""

    def __init__(self, master: Any, translator: Translator) -> None:
        super().__init__(master, fg_color="transparent")
        self._t = translator
        self._mode_var = ctk.StringVar(value="base")
        self._reference_var = ctk.StringVar(value="")
        self._speaker_var = ctk.StringVar(value="")
        self._instruct_var = ctk.StringVar(value="")
        self._mode_frames: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(self, text=self._t("synth.voice_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )
        segmented = ctk.CTkSegmentedButton(
            self,
            values=[self._t(f"synth.voice.{m}") for m in _MODES],
            variable=ctk.StringVar(value=self._t("synth.voice.base")),
            command=self._on_mode_changed,
        )
        segmented.grid(row=1, column=0, sticky="ew", pady=(0, SPACE_MD))
        self._segmented = segmented
        self._segmented_labels = [self._t(f"synth.voice.{m}") for m in _MODES]

        # Mode-specific input frames — created once, shown/hidden by
        # _on_mode_changed. This keeps the layout engine stable and
        # avoids rebuild flicker when switching modes.
        self._mode_frames["clone"] = self._build_clone_frame()
        self._mode_frames["custom"] = self._build_custom_frame()
        self._mode_frames["design"] = self._build_design_frame()

        for frame in self._mode_frames.values():
            frame.grid(row=2, column=0, sticky="ew")
            frame.grid_remove()

        self.grid_columnconfigure(0, weight=1)

    def _on_mode_changed(self, label: str) -> None:
        # Translate back from display label to internal mode key.
        idx = self._segmented_labels.index(label)
        mode = _MODES[idx]
        self._mode_var.set(mode)
        for name, frame in self._mode_frames.items():
            if name == mode:
                frame.grid()
            else:
                frame.grid_remove()

    def _build_clone_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(frame, text=self._t("synth.reference_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )
        entry = ctk.CTkEntry(frame, textvariable=self._reference_var, placeholder_text="ref.wav")
        entry.grid(row=1, column=0, sticky="ew", padx=(0, SPACE_SM))
        ctk.CTkButton(frame, text=self._t("synth.browse"), width=80, command=self._browse).grid(
            row=1, column=1
        )
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_custom_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(frame, text=self._t("synth.speaker_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )
        ctk.CTkEntry(frame, textvariable=self._speaker_var, placeholder_text="Vivian").grid(
            row=1, column=0, sticky="ew", pady=(0, SPACE_SM)
        )
        ctk.CTkLabel(frame, text=self._t("synth.instruct_label"), font=FONT_BODY).grid(
            row=2, column=0, sticky="w", pady=(SPACE_SM, SPACE_SM)
        )
        ctk.CTkEntry(frame, textvariable=self._instruct_var).grid(row=3, column=0, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_design_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(frame, text=self._t("synth.instruct_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )
        ctk.CTkEntry(
            frame,
            textvariable=self._instruct_var,
            placeholder_text="Speak in an incredulous tone…",
        ).grid(row=1, column=0, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title=self._t("synth.reference_label"),
            filetypes=[("Audio/JSON", "*.wav *.json"), ("All files", "*.*")],
        )
        if path:
            self._reference_var.set(path)

    def build_voice(self) -> Optional[Voice]:
        """Translate the current form state into a :class:`Voice`.

        Returns ``None`` if the user hasn't filled in required inputs;
        the caller shows a validation error instead of crashing.
        """
        mode = self._mode_var.get()
        if mode == "base":
            return Voice.base()
        if mode == "clone":
            path = self._reference_var.get().strip()
            if not path:
                return None
            return Voice.clone_file(Path(path))
        if mode == "custom":
            speaker = self._speaker_var.get().strip()
            if not speaker:
                return None
            return Voice.custom(speaker, instruct=self._instruct_var.get().strip())
        if mode == "design":
            instruct = self._instruct_var.get().strip()
            if not instruct:
                return None
            return Voice.design(instruct)
        return None
