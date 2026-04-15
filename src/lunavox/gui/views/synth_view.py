"""The main synthesize page.

Layout (top to bottom):
  * Text area
  * :class:`VoicePicker`
  * :class:`ParamSliderGroup`
  * Generate button + status line
  * :class:`StatsCard`
  * Action row (play / save / regenerate)

Runs the engine call on a background thread so the customtkinter
event loop stays responsive.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from lunavox.cli._config import ResolvedConfig
from lunavox.cli.synth_cmd import _write_wav
from lunavox.runtime import Engine, SynthesisParams, SynthesisResult, Voice

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
)
from ..widgets import FieldSpec, ParamSliderGroup, StatsCard, VoicePicker


class SynthView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, master: Any, config: ResolvedConfig, translator: Translator) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._last_result: Optional[SynthesisResult] = None
        self._worker: Optional[threading.Thread] = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- Left: text + voice + params + generate ---
        left = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_MD))
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text=self._t("synth.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        ctk.CTkLabel(left, text=self._t("synth.text_label"), font=FONT_BODY).grid(
            row=1, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_SM, SPACE_SM)
        )
        self._text = ctk.CTkTextbox(left, height=140, font=FONT_BODY)
        self._text.grid(row=2, column=0, sticky="ew", padx=SPACE_LG)
        self._text.insert("1.0", "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.")

        self._voice_picker = VoicePicker(left, translator=self._t)
        self._voice_picker.grid(row=3, column=0, sticky="ew", padx=SPACE_LG, pady=SPACE_MD)

        ctk.CTkLabel(left, text=self._t("synth.params_label"), font=FONT_HEADING).grid(
            row=4, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM)
        )
        self._params = ParamSliderGroup(
            left,
            fields=[
                FieldSpec("temperature", "Temperature", 0.0, 1.5, 0.05, self._config.temperature),
                FieldSpec("top_p", "Top-p", 0.0, 1.0, 0.05, self._config.top_p),
                FieldSpec("top_k", "Top-k", 1, 200, 1, self._config.top_k, cast=int),
                FieldSpec(
                    "repetition_penalty",
                    "Repetition penalty",
                    1.0,
                    2.0,
                    0.05,
                    self._config.repetition_penalty,
                ),
            ],
        )
        self._params.grid(row=5, column=0, sticky="ew", padx=SPACE_LG)

        self._generate_btn = ctk.CTkButton(
            left,
            text=self._t("synth.generate"),
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            height=42,
            font=FONT_HEADING,
            command=self._on_generate,
        )
        self._generate_btn.grid(row=6, column=0, sticky="ew", padx=SPACE_LG, pady=SPACE_MD)

        self._status = ctk.CTkLabel(left, text="", font=FONT_BODY)
        self._status.grid(row=7, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG))

        # --- Right: stats + actions ---
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        self._stats_card = StatsCard(right, translator=self._t)
        self._stats_card.grid(row=0, column=0, sticky="new", pady=(0, SPACE_MD))

        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew")
        actions.grid_columnconfigure((0, 1, 2), weight=1)

        self._play_btn = ctk.CTkButton(
            actions, text=self._t("synth.play"), command=self._on_play, state="disabled"
        )
        self._play_btn.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_SM))
        self._save_btn = ctk.CTkButton(
            actions, text=self._t("synth.save"), command=self._on_save, state="disabled"
        )
        self._save_btn.grid(row=0, column=1, sticky="ew", padx=SPACE_SM)
        self._regen_btn = ctk.CTkButton(
            actions,
            text=self._t("synth.regenerate"),
            command=self._on_generate,
            state="disabled",
        )
        self._regen_btn.grid(row=0, column=2, sticky="ew", padx=(SPACE_SM, 0))

    # --- actions -------------------------------------------------------

    def _on_generate(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        text = self._text.get("1.0", "end").strip()
        if not text:
            self._status.configure(text=self._t("synth.error") + ": empty text")
            return
        voice = self._voice_picker.build_voice()
        if voice is None:
            self._status.configure(text=self._t("synth.error") + ": voice inputs incomplete")
            return

        self._generate_btn.configure(state="disabled")
        self._status.configure(text=self._t("synth.generating"))
        self._worker = threading.Thread(
            target=self._run_synthesis,
            args=(text, voice),
            daemon=True,
        )
        self._worker.start()

    def _run_synthesis(self, text: str, voice: Voice) -> None:
        try:
            model_dir = self._config.project_root / "models" / self._config.model
            slider_values = self._params.values()
            params = SynthesisParams(
                temperature=float(slider_values["temperature"]),
                top_p=float(slider_values["top_p"]),
                top_k=int(slider_values["top_k"]),
                repetition_penalty=float(slider_values["repetition_penalty"]),
                n_threads=self._config.n_threads,
                language_id=self._config.language_id,
            )
            with Engine(model_dir, n_threads=self._config.n_threads) as engine:
                result = engine.synthesize(text, voice=voice, params=params)
            # Bounce back onto the Tk main loop so widgets update on the
            # thread that owns them — calling configure() from a
            # background thread is undefined behaviour in tcl/tk.
            self.after(0, self._on_synthesis_done, result)
        except Exception as err:
            self.after(0, self._on_synthesis_error, err)

    def _on_synthesis_done(self, result: SynthesisResult) -> None:
        self._last_result = result
        self._stats_card.update_stats(result.stats)
        self._status.configure(
            text=f"RTF {result.stats.rtf:.3f}  •  "
            f"{result.stats.audio_duration_ms / 1000:.2f}s  •  "
            f"total {result.stats.t_total_ms} ms"
        )
        self._generate_btn.configure(state="normal")
        for btn in (self._play_btn, self._save_btn, self._regen_btn):
            btn.configure(state="normal")

    def _on_synthesis_error(self, err: Exception) -> None:
        self._generate_btn.configure(state="normal")
        self._status.configure(text=f"{self._t('synth.error')}: {err}")

    def _on_play(self) -> None:
        if self._last_result is None:
            return
        try:
            import pygame  # pyright: ignore[reportMissingImports]
        except ImportError:
            self._status.configure(text="pygame missing — install lunavox[gui]")
            return
        # Write to a temp WAV and let pygame own playback — simpler
        # than juggling raw PCM buffers with multiple backends.
        tmp = self._config.project_root / "output" / "_gui_preview.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(tmp, self._last_result.audio, self._last_result.sample_rate)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(tmp))
        pygame.mixer.music.play()

    def _on_save(self) -> None:
        if self._last_result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="lunavox.wav",
        )
        if path:
            _write_wav(Path(path), self._last_result.audio, self._last_result.sample_rate)
            self._status.configure(text=f"Saved → {path}")
