"""A single widget that renders the voice inputs for the active model.

Each model type supports exactly one :class:`Voice` mode (the engine's
``supports_mode`` rejects every other combination): base → clone,
custom → custom, design → design. The picker therefore skips any
mode-selection UI and just builds the single applicable form — a
reference-audio picker, a speaker dropdown, or an instruction field.
Inside Custom, the speaker id is a dropdown sourced from
``model_profile.raw["speaker_names"]``. Style-instruction fields are
hidden when ``instruct_support`` is false (that is, for the 0.6B
custom model); when present they are paired with a curated preset
dropdown that populates the entry on selection.

The picker exposes :meth:`set_model_profile` so the parent Synthesize
view can swap profiles when the user picks a different installed
model, without recreating the widget tree above.
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from lunavox.core.synth import VoiceSpec
from lunavox.model.profile import ModelProfile

from ..i18n import Translator
from ..theme import FONT_BODY, SPACE_MD, SPACE_SM, TEXT_MUTED
from ..voice_presets import custom_presets, design_presets


def _modes_for(profile: Optional[ModelProfile]) -> tuple[str, ...]:
    if profile is None:
        return ()
    mtype = (profile.model_type or "").lower()
    if mtype == "base":
        return ("clone",)
    if mtype == "custom":
        return ("custom",)
    if mtype == "design":
        return ("design",)
    return ()


class VoicePicker(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    """Renders the voice inputs for the active model's single mode.

    ``model_profile`` is optional so the widget can render a disabled
    placeholder when no model is installed yet.
    """

    def __init__(
        self,
        master: Any,
        translator: Translator,
        model_profile: Optional[ModelProfile] = None,
        ref_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._t = translator
        self._model_profile: Optional[ModelProfile] = model_profile
        self._ref_dir: Optional[Path] = ref_dir

        # State vars survive rebuilds so the user's typed reference path
        # and instruct text carry over when they switch models.
        self._mode_var = ctk.StringVar(value="clone")
        self._reference_var = ctk.StringVar(value="")
        self._speaker_var = ctk.StringVar(value="")
        self._instruct_var = ctk.StringVar(value="")

        self._mode_frames: dict[str, Any] = {}
        self._active_modes: tuple[str, ...] = ()

        self.grid_columnconfigure(0, weight=1)
        self._rebuild()

    # --- public API ----------------------------------------------------

    def set_model_profile(self, profile: Optional[ModelProfile]) -> None:
        """Swap the active model and rebuild the applicable tabs."""
        self._model_profile = profile
        self._rebuild()

    def build_voice_spec(self) -> Optional[VoiceSpec]:
        """Translate the current form state into a :class:`VoiceSpec`.

        Returns ``None`` if the user hasn't filled in required inputs or
        no model is loaded; the caller shows a validation error and skips
        resolving the spec into a :class:`Voice`.

        Returning the transport-agnostic spec (instead of a runtime
        :class:`Voice`) means the view decides when and how to run
        :func:`lunavox.core.synth.resolve_voice` — typically on a
        background thread together with the engine construction — and
        can catch :class:`~lunavox.core.synth.VoiceResolutionError`
        with its own UX affordance.
        """
        if not self._active_modes:
            return None
        mode = self._mode_var.get()
        if mode == "clone":
            path = self._reference_var.get().strip()
            if not path:
                return None
            return VoiceSpec.clone(path)
        if mode == "custom":
            speaker = self._speaker_var.get().strip()
            if not speaker:
                return None
            instruct = (
                self._instruct_var.get().strip() if self._instruct_support() else ""
            )
            return VoiceSpec.custom(speaker, instruct=instruct)
        if mode == "design":
            instruct = self._instruct_var.get().strip()
            if not instruct:
                return None
            return VoiceSpec.design(instruct)
        return VoiceSpec.base()

    # --- internals -----------------------------------------------------

    def _instruct_support(self) -> bool:
        return bool(self._model_profile and self._model_profile.instruct_support)

    def _speaker_names(self) -> list[str]:
        if not self._model_profile:
            return []
        names = self._model_profile.raw.get("speaker_names") or []
        return [str(n) for n in names]

    def _rebuild(self) -> None:
        for child in list(self.winfo_children()):
            child.destroy()
        self._mode_frames.clear()

        modes = _modes_for(self._model_profile)
        self._active_modes = modes
        if not modes:
            ctk.CTkLabel(
                self,
                text=self._t("synth.no_models"),
                font=FONT_BODY,
                text_color=TEXT_MUTED,
            ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_MD))
            return

        # Each model type has exactly one applicable mode, so pin the
        # mode var and render that mode's frame directly — no tabs.
        mode = modes[0]
        self._mode_var.set(mode)

        builders = {
            "clone": self._build_clone_frame,
            "custom": self._build_custom_frame,
            "design": self._build_design_frame,
        }
        frame = builders[mode]()
        frame.grid(row=0, column=0, sticky="ew")
        self._mode_frames[mode] = frame

    def _list_ref_files(self) -> list[str]:
        if self._ref_dir is None or not self._ref_dir.exists():
            return []
        return sorted(
            p.name
            for p in self._ref_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".json"}
        )

    def _build_clone_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text=self._t("synth.ref_preset_label"), font=FONT_BODY).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, SPACE_SM)
        )

        custom_label = self._t("synth.ref_custom")
        ref_files = self._list_ref_files()
        preset_values = [custom_label, *ref_files]

        # Seed the dropdown from the current reference_var so the user's
        # prior selection survives mode/model rebuilds.
        current = self._reference_var.get().strip()
        initial = custom_label
        if current and self._ref_dir is not None:
            try:
                as_path = Path(current)
                if as_path.parent.resolve() == self._ref_dir.resolve() and as_path.name in ref_files:
                    initial = as_path.name
            except Exception:  # noqa: BLE001 — bad path → just fall back to custom
                pass

        preset_var = ctk.StringVar(value=initial)
        ctk.CTkOptionMenu(
            frame,
            values=preset_values,
            variable=preset_var,
            command=lambda v: self._on_ref_preset(v, custom_label),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, SPACE_SM))

        ctk.CTkLabel(frame, text=self._t("synth.reference_label"), font=FONT_BODY).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(SPACE_SM, SPACE_SM)
        )
        entry = ctk.CTkEntry(
            frame, textvariable=self._reference_var, placeholder_text="ref.wav"
        )
        entry.grid(row=3, column=0, sticky="ew", padx=(0, SPACE_SM))
        ctk.CTkButton(
            frame,
            text=self._t("synth.browse"),
            width=80,
            command=lambda: self._browse(preset_var, custom_label),
        ).grid(row=3, column=1)
        return frame

    def _on_ref_preset(self, selected: str, custom_label: str) -> None:
        if selected == custom_label or self._ref_dir is None:
            # Let the user type or browse; don't clobber what they had.
            return
        self._reference_var.set(str(self._ref_dir / selected))

    def _build_custom_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text=self._t("synth.speaker_label"), font=FONT_BODY).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_SM)
        )

        speakers = self._speaker_names()
        if speakers:
            if self._speaker_var.get() not in speakers:
                self._speaker_var.set(speakers[0])
            ctk.CTkOptionMenu(
                frame, values=speakers, variable=self._speaker_var
            ).grid(row=1, column=0, sticky="ew", pady=(0, SPACE_SM))
        else:
            ctk.CTkEntry(
                frame, textvariable=self._speaker_var, placeholder_text="Vivian"
            ).grid(row=1, column=0, sticky="ew", pady=(0, SPACE_SM))

        if self._instruct_support():
            self._add_instruct_controls(frame, start_row=2, presets=custom_presets(self._t.lang))
        return frame

    def _build_design_frame(self) -> Any:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        # Design mode always needs instruct; gate symmetrically just in case.
        if self._instruct_support():
            self._add_instruct_controls(
                frame, start_row=0, presets=design_presets(self._t.lang)
            )
        else:
            ctk.CTkLabel(
                frame,
                text=self._t("synth.no_models"),
                font=FONT_BODY,
                text_color=TEXT_MUTED,
            ).grid(row=0, column=0, sticky="w")
        return frame

    def _add_instruct_controls(
        self,
        frame: Any,
        start_row: int,
        presets: list[tuple[str, str]],
    ) -> None:
        # Preset dropdown: selecting a non-empty preset writes its
        # instruction into the entry var so the user can still tweak it.
        ctk.CTkLabel(frame, text=self._t("synth.preset_label"), font=FONT_BODY).grid(
            row=start_row, column=0, sticky="w", pady=(SPACE_SM, SPACE_SM)
        )
        preset_labels = [p[0] for p in presets]
        preset_map = {p[0]: p[1] for p in presets}
        preset_var = ctk.StringVar(value=preset_labels[0])

        def _on_preset(label: str) -> None:
            value = preset_map.get(label, "")
            if value:
                self._instruct_var.set(value)

        ctk.CTkOptionMenu(
            frame, values=preset_labels, variable=preset_var, command=_on_preset
        ).grid(row=start_row + 1, column=0, sticky="ew", pady=(0, SPACE_SM))

        ctk.CTkLabel(frame, text=self._t("synth.instruct_label"), font=FONT_BODY).grid(
            row=start_row + 2, column=0, sticky="w", pady=(SPACE_SM, SPACE_SM)
        )
        ctk.CTkEntry(
            frame,
            textvariable=self._instruct_var,
            placeholder_text="Speak in an incredulous tone…",
        ).grid(row=start_row + 3, column=0, sticky="ew")

    def _browse(
        self,
        preset_var: Optional[Any] = None,
        custom_label: Optional[str] = None,
    ) -> None:
        path = filedialog.askopenfilename(
            title=self._t("synth.reference_label"),
            filetypes=[("Audio/JSON", "*.wav *.json"), ("All files", "*.*")],
        )
        if path:
            self._reference_var.set(path)
            # Browsing means "custom" — reset the dropdown so the UI
            # doesn't imply the chosen file is a preset ref.
            if preset_var is not None and custom_label is not None:
                preset_var.set(custom_label)
