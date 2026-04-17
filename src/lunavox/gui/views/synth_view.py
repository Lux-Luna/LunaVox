"""The main synthesize page.

Layout:
  * Left column — model dropdown, text area, :class:`VoicePicker`
    (rebuilt when the model changes), generate button + status line.
  * Right column — stats card, play/save/regenerate actions, and a
    collapsible :class:`ParamSliderGroup` below them.

"Generate" streams audio through :class:`SynthesisPipeline.synthesize_stream`
and pipes each chunk straight into a :mod:`sounddevice` output stream on
a background thread, so playback begins as soon as the first chunk is
ready (TTFB measured in wall-clock, not C-engine-reported). The Play
button is a pure replay of the accumulated buffer — never used for the
first listen.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]
import numpy as np

from lunavox.cli._config import ResolvedConfig
from lunavox.core.synth import (
    SynthesisPipeline,
    VoiceResolutionError,
    VoiceSpec,
    f32_to_wav,
    resolve_voice,
)
from lunavox.model import all_models
from lunavox.model.profile import ModelProfile
from lunavox.runtime import SynthesisParams, SynthesisStats

from ..engine_session import EngineSession
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


def _installed_model_keys(project_root: Path) -> list[str]:
    models_dir = project_root / "models"
    if not models_dir.exists():
        return []
    keys: list[str] = []
    for spec in all_models():
        local = models_dir / spec.name
        if local.exists() and any(local.iterdir()):
            keys.append(spec.name)
    return keys


def _safe_load_profile(project_root: Path, model_name: str) -> Optional[ModelProfile]:
    try:
        return ModelProfile.load(project_root / "models" / model_name)
    except Exception:  # noqa: BLE001 — profile missing/corrupt = fall back to None
        return None


class SynthView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(
        self,
        master: Any,
        config: ResolvedConfig,
        translator: Translator,
        engine_session: EngineSession,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        self._session = engine_session
        self._last_audio: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._worker: Optional[threading.Thread] = None
        # Held while an async session.load() is in flight; we disable
        # the competing UI controls (Generate, dropdown, checkbox)
        # for the duration so the user can't stack ops on top of a
        # native-library load that can't be cancelled.
        self._load_worker: Optional[threading.Thread] = None
        # Set by the main thread from _on_generate to ask the running
        # worker to bail out mid-stream (on Regenerate). The worker
        # checks it between chunks and also relies on _active_stream
        # being aborted from the UI thread for instant silence.
        self._cancel = threading.Event()
        # Held while a live-preview sounddevice stream is open; the UI
        # thread calls .abort() on it during Regenerate to cut playback
        # without waiting for the worker to notice the cancel flag.
        self._active_stream: Any = None
        # Populated in _build() when there's at least one installed
        # model; left as None otherwise so guards can short-circuit.
        self._model_menu: Any = None
        self._persistent_var: Any = None
        self._persistent_cb: Any = None

        installed = _installed_model_keys(self._config.project_root)
        # Prefer the configured model if it's installed; else pick the
        # first installed one; else leave empty.
        if self._config.model in installed:
            self._current_model = self._config.model
        elif installed:
            self._current_model = installed[0]
            self._config.model = installed[0]
        else:
            self._current_model = ""
        self._installed_models = installed

        self._build()

    def _build(self) -> None:
        # Left gets roughly twice the right column so the text area and
        # voice picker have room to breathe; the right column only has
        # the stats card, three action buttons, and the (usually
        # collapsed) slider group.
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Left: model + text + voice + generate ---
        left = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_MD))
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text=self._t("synth.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        # Model picker row.
        ctk.CTkLabel(left, text=self._t("synth.model_label"), font=FONT_BODY).grid(
            row=1, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_SM, SPACE_SM)
        )
        if self._installed_models:
            # Dropdown + Pre-load checkbox share one row. The dropdown
            # takes the remaining width; the checkbox sits on the right
            # and controls the EngineSession cache directly — ticking
            # it kicks off a background engine load so subsequent
            # Generates skip the cold-start penalty.
            model_row = ctk.CTkFrame(left, fg_color="transparent")
            model_row.grid(row=2, column=0, sticky="ew", padx=SPACE_LG)
            model_row.grid_columnconfigure(0, weight=1)

            self._model_var = ctk.StringVar(value=self._current_model)
            self._model_menu = ctk.CTkOptionMenu(
                model_row,
                values=self._installed_models,
                variable=self._model_var,
                command=self._on_model_changed,
            )
            self._model_menu.grid(row=0, column=0, sticky="ew")

            # Initial check state mirrors the session so that when the
            # user flips tabs and comes back, a still-loaded model
            # shows its box ticked.
            model_dir = self._config.project_root / "models" / self._current_model
            initial_loaded = self._session.is_loaded_for(
                model_dir, self._config.n_threads
            )
            self._persistent_var = ctk.BooleanVar(value=initial_loaded)
            self._persistent_cb = ctk.CTkCheckBox(
                model_row,
                text=self._t("synth.persistent_label"),
                variable=self._persistent_var,
                command=self._on_persistent_toggled,
                font=FONT_BODY,
            )
            self._persistent_cb.grid(row=0, column=1, padx=(SPACE_MD, 0))
        else:
            self._model_var = ctk.StringVar(value="")
            self._model_menu = None
            self._persistent_var = None
            self._persistent_cb = None
            ctk.CTkLabel(
                left,
                text=self._t("synth.no_models"),
                font=FONT_BODY,
            ).grid(row=2, column=0, sticky="w", padx=SPACE_LG)

        ctk.CTkLabel(left, text=self._t("synth.text_label"), font=FONT_BODY).grid(
            row=3, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM)
        )
        self._text = ctk.CTkTextbox(left, height=140, font=FONT_BODY)
        self._text.grid(row=4, column=0, sticky="ew", padx=SPACE_LG)
        self._text.insert("1.0", "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.")

        initial_profile = (
            _safe_load_profile(self._config.project_root, self._current_model)
            if self._current_model
            else None
        )
        self._voice_picker = VoicePicker(
            left,
            translator=self._t,
            model_profile=initial_profile,
            ref_dir=self._config.project_root / "ref",
        )
        self._voice_picker.grid(row=5, column=0, sticky="ew", padx=SPACE_LG, pady=SPACE_MD)

        self._generate_btn = ctk.CTkButton(
            left,
            text=self._t("synth.generate"),
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            height=42,
            font=FONT_HEADING,
            command=self._on_generate,
            state="normal" if self._installed_models else "disabled",
        )
        self._generate_btn.grid(row=6, column=0, sticky="ew", padx=SPACE_LG, pady=SPACE_MD)

        self._status = ctk.CTkLabel(left, text="", font=FONT_BODY)
        self._status.grid(row=7, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG))
        if not self._installed_models:
            self._status.configure(text=self._t("synth.no_models"))

        # --- Right: stats + actions + advanced params ---
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

        # Advanced parameters live under the stats/actions column.
        # Collapsed by default — most runs use profile defaults, so
        # keep the sliders out of the way until the user opts in.
        self._params_expanded = False
        self._params_toggle = ctk.CTkButton(
            right,
            text=self._t("synth.advanced_expand"),
            anchor="w",
            fg_color="transparent",
            hover_color=PRIMARY_HOVER,
            text_color=("#1A202C", "#E2E8F0"),
            font=FONT_HEADING,
            command=self._toggle_params,
        )
        self._params_toggle.grid(row=2, column=0, sticky="ew", pady=(SPACE_MD, SPACE_SM))
        self._params = ParamSliderGroup(
            right,
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
        self._params.grid(row=3, column=0, sticky="ew")
        self._params.grid_remove()

    # --- actions -------------------------------------------------------

    def _toggle_params(self) -> None:
        self._params_expanded = not self._params_expanded
        if self._params_expanded:
            self._params.grid()
            self._params_toggle.configure(text=self._t("synth.advanced_collapse"))
        else:
            self._params.grid_remove()
            self._params_toggle.configure(text=self._t("synth.advanced_expand"))

    def _on_model_changed(self, model_name: str) -> None:
        self._current_model = model_name
        self._config.model = model_name
        profile = _safe_load_profile(self._config.project_root, model_name)
        self._voice_picker.set_model_profile(profile)
        # The cache (if any) was for the previous model — release it
        # and force the checkbox back to unchecked. The user has to
        # tick Pre-load again to load the new model.
        self._session.unload()
        if self._persistent_var is not None:
            self._persistent_var.set(False)

    def _on_persistent_toggled(self) -> None:
        """React to the user ticking / unticking the Pre-load checkbox.

        * Ticked → dispatch a worker thread that calls session.load().
          The dropdown, Generate and the checkbox itself are disabled
          for the duration so the load can't be stacked on top of a
          Generate or a model switch.
        * Unticked → unload synchronously (cheap: just frees the C
          handle) and clear the status line.
        """
        if self._persistent_var is None:
            return
        want = bool(self._persistent_var.get())
        if want:
            if self._load_worker is not None and self._load_worker.is_alive():
                return  # already loading
            self._set_load_ui(loading=True)
            self._status.configure(text=self._t("synth.persistent_loading"))
            self._load_worker = threading.Thread(
                target=self._run_load, daemon=True
            )
            self._load_worker.start()
        else:
            self._session.unload()
            self._status.configure(text="")

    def _set_load_ui(self, *, loading: bool) -> None:
        """Toggle the controls that must not race with a session.load()."""
        state = "disabled" if loading else "normal"
        self._generate_btn.configure(state=state)
        if self._model_menu is not None:
            self._model_menu.configure(state=state)
        if self._persistent_cb is not None:
            self._persistent_cb.configure(state=state)

    def _run_load(self) -> None:
        try:
            model_dir = self._config.project_root / "models" / self._config.model
            self._session.load(model_dir, self._config.n_threads)
        except Exception as err:  # noqa: BLE001 — surfaced on status line
            self.after(0, self._on_load_done, err)
            return
        self.after(0, self._on_load_done, None)

    def _on_load_done(self, err: Optional[Exception]) -> None:
        self._set_load_ui(loading=False)
        if err is None:
            self._status.configure(text=self._t("synth.persistent_loaded"))
            # Regenerate stays in sync with whatever prior state it had
            # (enabled if there's an old buffer, disabled otherwise).
            self._regen_btn.configure(
                state="normal" if self._last_audio is not None else "disabled"
            )
        else:
            # Load failed — revert the visual so it matches reality.
            if self._persistent_var is not None:
                self._persistent_var.set(False)
            self._status.configure(
                text=f"{self._t('synth.persistent_load_failed')}: {err}"
            )

    def _on_generate(self) -> None:
        if not self._current_model:
            self._status.configure(text=self._t("synth.no_models"))
            return
        text = self._text.get("1.0", "end").strip()
        if not text:
            self._status.configure(text=self._t("synth.error") + ": empty text")
            return
        spec = self._voice_picker.build_voice_spec()
        if spec is None:
            self._status.configure(text=self._t("synth.error") + ": voice inputs incomplete")
            return

        # Regenerate pressed while a worker is still streaming: cut the
        # current playback and join before the new run starts, so
        # writes to the next OutputStream can't race with the old one.
        if self._worker and self._worker.is_alive():
            self._cancel.set()
            stream = self._active_stream
            if stream is not None:
                # device may already be closing — best-effort abort.
                with contextlib.suppress(Exception):
                    stream.abort()
            self._worker.join(timeout=1.0)
        self._cancel.clear()

        self._generate_btn.configure(state="disabled")
        self._play_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        # Freeze the Pre-load box during a run so the user can't kick
        # off a concurrent model reload that would also contend for
        # CPU/GPU. Re-enabled on done or error.
        if self._persistent_cb is not None:
            self._persistent_cb.configure(state="disabled")
        # Regenerate stays clickable mid-run so the user can interrupt.
        self._status.configure(text=self._t("synth.generating"))

        start_ts = time.monotonic()
        self._worker = threading.Thread(
            target=self._run_synthesis,
            args=(text, spec, start_ts),
            daemon=True,
        )
        self._worker.start()

    def _run_synthesis(self, text: str, spec: VoiceSpec, start_ts: float) -> None:
        try:
            model_dir = self._config.project_root / "models" / self._config.model
            slider_values = self._params.values()
            params = SynthesisParams.from_overrides(
                temperature=float(slider_values["temperature"]),
                top_p=float(slider_values["top_p"]),
                top_k=int(slider_values["top_k"]),
                repetition_penalty=float(slider_values["repetition_penalty"]),
                n_threads=self._config.n_threads,
                language_id=self._config.language_id,
            )
            try:
                voice = resolve_voice(spec)
            except VoiceResolutionError as err:
                self.after(0, self._on_synthesis_error, err)
                return

            try:
                import sounddevice as sd
            except (ImportError, OSError) as err:
                self.after(
                    0,
                    self._on_synthesis_error,
                    RuntimeError(
                        "sounddevice unavailable — install the 'lunavox' package "
                        f"(PortAudio is required for live playback): {err}"
                    ),
                )
                return

            chunks: list[np.ndarray] = []
            ttfb_ms: Optional[float] = None
            final_stats: Optional[SynthesisStats] = None
            sr: Optional[int] = None
            out_stream: Any = None

            try:
                with self._session.acquire(
                    model_dir, n_threads=self._config.n_threads
                ) as engine:
                    pipeline = SynthesisPipeline(engine)
                    for chunk in pipeline.synthesize_stream(
                        text, voice=voice, params=params
                    ):
                        if self._cancel.is_set():
                            break
                        if len(chunk.audio) > 0:
                            if out_stream is None:
                                sr = chunk.sample_rate
                                out_stream = sd.OutputStream(
                                    samplerate=sr, channels=1, dtype="float32"
                                )
                                out_stream.start()
                                self._active_stream = out_stream
                                ttfb_ms = (time.monotonic() - start_ts) * 1000
                                self.after(0, self._on_first_audio, ttfb_ms)
                            out_stream.write(chunk.audio)
                            chunks.append(chunk.audio)
                        if chunk.is_last:
                            final_stats = chunk.stats
            finally:
                if out_stream is not None:
                    try:
                        out_stream.stop()
                        out_stream.close()
                    except Exception:  # noqa: BLE001 — best-effort teardown
                        pass
                self._active_stream = None

            if self._cancel.is_set():
                # Regenerate cut us off mid-run. The new worker owns
                # the UI now — don't clobber its state with ours.
                return

            full_audio = (
                np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            )
            self.after(
                0, self._on_synthesis_done, full_audio, sr, final_stats, ttfb_ms
            )
        except Exception as err:  # noqa: BLE001 — surfaced to user via status line
            self.after(0, self._on_synthesis_error, err)

    def _on_first_audio(self, ttfb_ms: float) -> None:
        # Surfacing TTFB on the status line the moment playback starts
        # gives the user an instant answer to "did the model respond?"
        # without waiting for the stats card to populate on is_last.
        self._status.configure(text=f"Playing…  TTFB {ttfb_ms:.0f} ms")

    def _on_synthesis_done(
        self,
        audio: np.ndarray,
        sr: Optional[int],
        stats: Optional[SynthesisStats],
        ttfb_ms: Optional[float],
    ) -> None:
        if audio.size > 0 and sr is not None:
            self._last_audio = audio
            self._last_sr = sr
        self._stats_card.update_stats(stats, ttfb_ms=ttfb_ms)
        if stats is not None:
            tail = f"  •  TTFB {ttfb_ms:.0f} ms" if ttfb_ms is not None else ""
            self._status.configure(
                text=f"RTF {stats.rtf:.3f}  •  "
                f"{stats.audio_duration_ms / 1000:.2f}s  •  "
                f"total {stats.t_total_ms} ms" + tail
            )
        self._generate_btn.configure(state="normal")
        playable = self._last_audio is not None
        self._play_btn.configure(state="normal" if playable else "disabled")
        self._save_btn.configure(state="normal" if playable else "disabled")
        self._regen_btn.configure(state="normal")
        if self._persistent_cb is not None:
            self._persistent_cb.configure(state="normal")

    def _on_synthesis_error(self, err: Exception) -> None:
        self._generate_btn.configure(state="normal")
        # Leave Play / Save in whatever state the prior successful run
        # set — the error doesn't invalidate the last good buffer.
        if self._last_audio is not None:
            self._regen_btn.configure(state="normal")
        if self._persistent_cb is not None:
            self._persistent_cb.configure(state="normal")
        self._status.configure(text=f"{self._t('synth.error')}: {err}")

    def _on_play(self) -> None:
        # Replay-only: never reached before _on_synthesis_done populates
        # the buffer (the button is disabled until then).
        if self._last_audio is None or self._last_sr is None:
            return
        import pygame

        # Write to a temp WAV and let pygame own playback — simpler
        # than juggling raw PCM buffers with multiple backends.
        # pygame.mixer.music keeps an OS-level handle on the loaded
        # file, which on Windows blocks reopening the same path for
        # writing. Stop + unload before rewriting so the second Play
        # click doesn't trip a PermissionError.
        tmp = self._config.project_root / "output" / "_gui_preview.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.stop()
        # unload() exists on pygame >= 2.0; older builds release the
        # handle when a new file is loaded.
        with contextlib.suppress(AttributeError, pygame.error):
            pygame.mixer.music.unload()
        tmp.write_bytes(f32_to_wav(self._last_audio, self._last_sr))
        pygame.mixer.music.load(str(tmp))
        pygame.mixer.music.play()

    def _on_save(self) -> None:
        if self._last_audio is None or self._last_sr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="lunavox.wav",
        )
        if path:
            Path(path).write_bytes(f32_to_wav(self._last_audio, self._last_sr))
            self._status.configure(text=f"Saved → {path}")
