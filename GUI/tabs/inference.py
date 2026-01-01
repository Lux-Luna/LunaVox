"""
LunaVox Inference Tab - Voice Synthesis Interface
Supports both Persona mode (default) and Reference Audio mode.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import shutil
import os
import json
import tempfile

from i18n import get_text
from theme import COLORS, FONTS, SPACING, get_text_widget_config
from widgets import LatencyDisplay, AudioOutputPanel, StatusIndicator

# Language code to i18n key mapping
LANG_KEY_MAP = {
    "en": "lang_en",
    "zh": "lang_zh",
    "ja": "lang_ja",
    "auto": "lang_auto",
}


class InferenceTab(ttk.Frame):
    """TTS Inference tab with persona/reference audio mode toggle."""
    
    def __init__(self, parent, app):
        super().__init__(parent, padding=SPACING["lg"])
        self.app = app
        self.configure(style="TFrame")
        
        # State
        self.tts_mode = tk.StringVar(value="persona")  # "persona" or "reference"
        self.persona_paths = {}  # name -> path mapping
        self.last_output_path = None  # Track last generated audio
        self.last_audio_duration = None  # Track last audio duration
        self.auto_save = tk.BooleanVar(value=False)  # Auto-save toggle (default: off)
        self.temp_audio_path = None  # Temp file for unsaved audio
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the inference tab UI."""
        # Scrollable container
        canvas = tk.Canvas(self, bg=COLORS["bg_dark"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style="TFrame")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Build sections
        self._build_mode_section()
        self._build_model_section()
        self._build_persona_section()
        self._build_reference_section()
        self._build_synthesis_section()
        
        # Initial state
        self._on_mode_change()
        self._update_persona_list()
        self._update_character_list()
        
    def _build_mode_section(self):
        """Build the TTS mode toggle section."""
        mode_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "tts_mode"),
            padding=SPACING["md"]
        )
        mode_frame.pack(fill="x", pady=(0, SPACING["md"]))
        
        # Radio buttons for mode selection
        modes_container = ttk.Frame(mode_frame, style="Card.TFrame")
        modes_container.pack(fill="x")
        
        self.radio_persona = ttk.Radiobutton(
            modes_container,
            text=get_text(self.app.lang, "tts_mode_persona"),
            variable=self.tts_mode,
            value="persona",
            command=self._on_mode_change,
            style="Card.TRadiobutton"
        )
        self.radio_persona.pack(side="left", padx=SPACING["lg"])
        
        self.radio_reference = ttk.Radiobutton(
            modes_container,
            text=get_text(self.app.lang, "tts_mode_reference"),
            variable=self.tts_mode,
            value="reference",
            command=self._on_mode_change,
            style="Card.TRadiobutton"
        )
        self.radio_reference.pack(side="left", padx=SPACING["lg"])
        
    def _build_model_section(self):
        """Build the model/character selection section."""
        self.model_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "character"),
            padding=SPACING["md"]
        )
        self.model_frame.pack(fill="x", pady=(0, SPACING["md"]))
        
        # Version selection
        row1 = ttk.Frame(self.model_frame, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, SPACING["sm"]))
        
        ttk.Label(row1, text=get_text(self.app.lang, "version"), style="Card.TLabel").pack(side="left")
        self.version_cb = ttk.Combobox(row1, values=["v2", "v2_pro_plus"], state="readonly", width=15)
        self.version_cb.set("v2")
        self.version_cb.pack(side="left", padx=SPACING["sm"])
        self.version_cb.bind("<<ComboboxSelected>>", lambda e: self._update_character_list())
        
        ttk.Label(row1, text=get_text(self.app.lang, "character"), style="Card.TLabel").pack(side="left", padx=(SPACING["lg"], 0))
        self.char_cb = ttk.Combobox(row1, state="readonly", width=25)
        self.char_cb.pack(side="left", padx=SPACING["sm"])
        
        # Buttons
        row2 = ttk.Frame(self.model_frame, style="Card.TFrame")
        row2.pack(fill="x", pady=SPACING["sm"])
        
        self.btn_load = ttk.Button(row2, text=get_text(self.app.lang, "load_model"), command=self._load_model)
        self.btn_load.pack(side="left", padx=(0, SPACING["sm"]))
        
        self.btn_unload = ttk.Button(row2, text=get_text(self.app.lang, "unload_model"), command=self._unload_model)
        self.btn_unload.pack(side="left")

        # Loading status indicator
        self.load_indicator = ttk.Label(row2, text="", style="Muted.TLabel")
        self.load_indicator.pack(side="left", padx=SPACING["lg"])
        
    def _build_persona_section(self):
        """Build the persona selection section (shown in persona mode)."""
        self.persona_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "select_persona"),
            padding=SPACING["md"]
        )
        
        row = ttk.Frame(self.persona_frame, style="Card.TFrame")
        row.pack(fill="x")
        
        self.persona_cb = ttk.Combobox(row, state="readonly", width=30)
        self.persona_cb.pack(side="left", fill="x", expand=True)
        self.persona_cb.bind("<<ComboboxSelected>>", self._on_persona_selected)
        
        self.btn_load_persona = ttk.Button(
            row,
            text=get_text(self.app.lang, "load_persona"),
            command=self._load_persona,
            style="Primary.TButton"
        )
        self.btn_load_persona.pack(side="right", padx=(SPACING["sm"], 0))
        
        # Persona status row
        status_row = ttk.Frame(self.persona_frame, style="Card.TFrame")
        status_row.pack(fill="x", pady=(SPACING["sm"], 0))
        
        # Persona language display
        self.persona_lang_label = ttk.Label(
            status_row,
            text="",
            style="Muted.TLabel"
        )
        self.persona_lang_label.pack(side="left")
        
        # Persona status indicator
        self.persona_status = StatusIndicator(status_row)
        self.persona_status.pack(side="right")
        self._reset_persona_status()
        
        # Hint label
        hint_row = ttk.Frame(self.persona_frame, style="Card.TFrame")
        hint_row.pack(fill="x", pady=(SPACING["sm"], 0))
        
        self.persona_hint_label = ttk.Label(
            hint_row,
            text=get_text(self.app.lang, "hint_lang_match"),
            style="Muted.TLabel",
            font=(FONTS["family"], FONTS["size_sm"])
        )
        self.persona_hint_label.pack(anchor="w")
        
    def _build_reference_section(self):
        """Build the reference audio section (shown in reference mode)."""
        self.ref_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "ref_audio"),
            padding=SPACING["md"]
        )
        
        # Language selection
        row1 = ttk.Frame(self.ref_frame, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, SPACING["sm"]))
        
        ttk.Label(row1, text=get_text(self.app.lang, "ref_lang"), style="Card.TLabel").pack(side="left")
        self.ref_lang_cb = ttk.Combobox(row1, values=["en", "zh", "ja"], state="readonly", width=10)
        self.ref_lang_cb.set("ja")
        self.ref_lang_cb.pack(side="left", padx=SPACING["sm"])
        self.ref_lang_cb.bind("<<ComboboxSelected>>", lambda e: self._update_preset_audio_list())
        
        ttk.Label(row1, text=get_text(self.app.lang, "preset_audio"), style="Card.TLabel").pack(side="left", padx=(SPACING["lg"], 0))
        self.preset_cb = ttk.Combobox(row1, state="readonly", width=25)
        self.preset_cb.pack(side="left", padx=SPACING["sm"])
        self.preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)
        
        # Audio file path
        row2 = ttk.Frame(self.ref_frame, style="Card.TFrame")
        row2.pack(fill="x", pady=(0, SPACING["sm"]))
        
        ttk.Label(row2, text=get_text(self.app.lang, "ref_audio"), style="Card.TLabel").pack(side="left")
        self.ref_audio_entry = ttk.Entry(row2)
        self.ref_audio_entry.pack(side="left", fill="x", expand=True, padx=SPACING["sm"])
        ttk.Button(row2, text=get_text(self.app.lang, "browse"), command=self._browse_ref_audio).pack(side="right")
        
        # Reference text
        row3 = ttk.Frame(self.ref_frame, style="Card.TFrame")
        row3.pack(fill="x")
        
        ttk.Label(row3, text=get_text(self.app.lang, "ref_text"), style="Card.TLabel").pack(side="left")
        self.ref_text_entry = ttk.Entry(row3)
        self.ref_text_entry.pack(side="left", fill="x", expand=True, padx=SPACING["sm"])
        
        # Reference status row
        ref_status_row = ttk.Frame(self.ref_frame, style="Card.TFrame")
        ref_status_row.pack(fill="x", pady=(SPACING["sm"], 0))
        
        self.ref_status = StatusIndicator(ref_status_row)
        self.ref_status.pack(side="right")
        self._reset_ref_status()
        
        # Hint labels
        hints_frame = ttk.Frame(self.ref_frame, style="Card.TFrame")
        hints_frame.pack(fill="x", pady=(SPACING["sm"], 0))
        
        self.ref_hint_lang = ttk.Label(
            hints_frame,
            text=get_text(self.app.lang, "hint_lang_match"),
            style="Muted.TLabel",
            font=(FONTS["family"], FONTS["size_sm"])
        )
        self.ref_hint_lang.pack(anchor="w")
        
        self.ref_hint_text = ttk.Label(
            hints_frame,
            text=get_text(self.app.lang, "hint_ref_text_accuracy"),
            style="Muted.TLabel",
            font=(FONTS["family"], FONTS["size_sm"])
        )
        self.ref_hint_text.pack(anchor="w", pady=(SPACING["xs"], 0))
        
        # Initialize preset list
        self.preset_audio_paths = {}
        self._update_preset_audio_list()
        
    def _build_synthesis_section(self):
        """Build the text input and synthesis section."""
        synth_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "target_text"),
            padding=SPACING["md"]
        )
        synth_frame.pack(fill="both", expand=True, pady=(0, SPACING["md"]))
        
        # Language selection
        lang_row = ttk.Frame(synth_frame, style="Card.TFrame")
        lang_row.pack(fill="x", pady=(0, SPACING["sm"]))
        
        ttk.Label(lang_row, text=get_text(self.app.lang, "target_lang"), style="Card.TLabel").pack(side="left")
        self.target_lang_cb = ttk.Combobox(lang_row, values=["en", "zh", "ja"], state="readonly", width=10)
        self.target_lang_cb.set("en")
        self.target_lang_cb.pack(side="left", padx=SPACING["sm"])
        
        # Text input
        text_config = get_text_widget_config()
        self.text_input = tk.Text(synth_frame, height=6, **text_config)
        self.text_input.pack(fill="both", expand=True, pady=SPACING["sm"])
        
        # Control buttons
        btn_row = ttk.Frame(synth_frame, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(SPACING["sm"], 0))
        
        self.btn_synth = ttk.Button(
            btn_row,
            text=get_text(self.app.lang, "synthesize"),
            command=self._synthesize,
            style="Primary.TButton"
        )
        self.btn_synth.pack(side="left", padx=(0, SPACING["sm"]))
        
        self.btn_stop = ttk.Button(
            btn_row,
            text=get_text(self.app.lang, "stop"),
            command=self._stop
        )
        self.btn_stop.pack(side="left")
        
        # Auto-save checkbox
        self.auto_save_cb = ttk.Checkbutton(
            btn_row,
            text=get_text(self.app.lang, "auto_save"),
            variable=self.auto_save,
            style="Card.TCheckbutton"
        )
        self.auto_save_cb.pack(side="right")

        
        # Latency display
        self.latency_display = LatencyDisplay(synth_frame)
        self.latency_display.pack(fill="x", pady=(SPACING["sm"], 0))
        
        # Audio output panel
        self.audio_panel = AudioOutputPanel(
            synth_frame,
            on_save=self._save_audio,
            on_play=self._play_audio
        )
        self.audio_panel.pack(fill="x", pady=(SPACING["sm"], 0))
        
    def _on_mode_change(self):
        """Handle TTS mode toggle."""
        mode = self.tts_mode.get()
        
        if mode == "persona":
            self.persona_frame.pack(fill="x", pady=(0, SPACING["md"]), after=self.model_frame)
            self.ref_frame.pack_forget()
        else:
            self.persona_frame.pack_forget()
            self.ref_frame.pack(fill="x", pady=(0, SPACING["md"]), after=self.model_frame)
            
    def _update_character_list(self):
        """Update the character model dropdown."""
        from main import REPO_ROOT
        version = self.version_cb.get()
        base_dir = REPO_ROOT / "CharacterData" / "model" / version
        
        if not base_dir.exists():
            self.char_cb['values'] = []
            return
            
        chars = [p.name for p in base_dir.iterdir() if p.is_dir()]
        self.char_cb['values'] = sorted(chars)
        if chars:
            self.char_cb.set(chars[0])
            
    def _update_persona_list(self):
        """Update the persona dropdown."""
        from main import REPO_ROOT
        persona_dir = REPO_ROOT / "CharacterData" / "character"
        
        if not persona_dir.exists():
            self.persona_cb['values'] = []
            return
            
        personas = []
        self.persona_paths = {}
        
        for p in persona_dir.iterdir():
            if p.is_dir():
                # Check if it has persona files
                if (p / "features.npz").exists() or (p / "metadata.json").exists():
                    personas.append(p.name)
                    self.persona_paths[p.name] = str(p)
                    
        self.persona_cb['values'] = sorted(personas)
        if personas:
            if "luna_en" in personas:
                self.persona_cb.set("luna_en")
            else:
                self.persona_cb.set(personas[0])
            # Trigger language display update
            self._on_persona_selected()

            
    def _update_preset_audio_list(self):
        """Update the preset audio dropdown based on selected language."""
        from main import REPO_ROOT
        lang_map = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
        lang_folder = lang_map.get(self.ref_lang_cb.get(), "Japanese")
        audio_dir = REPO_ROOT / "CharacterData" / "audio" / lang_folder
        
        if not audio_dir.exists():
            self.preset_cb['values'] = []
            self.preset_audio_paths = {}
            return
            
        wavs = list(audio_dir.glob("*.wav"))
        self.preset_cb['values'] = [w.name for w in wavs]
        self.preset_audio_paths = {w.name: str(w) for w in wavs}
        
    def _on_preset_selected(self, event):
        """Handle preset audio selection."""
        filename = self.preset_cb.get()
        path = self.preset_audio_paths.get(filename)
        if path:
            self.ref_audio_entry.delete(0, tk.END)
            self.ref_audio_entry.insert(0, path)
            self.ref_text_entry.delete(0, tk.END)
            self.ref_text_entry.insert(0, Path(path).stem)
            
    def _browse_ref_audio(self):
        """Browse for reference audio file."""
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self.ref_audio_entry.delete(0, tk.END)
            self.ref_audio_entry.insert(0, path)
            if not self.ref_text_entry.get():
                self.ref_text_entry.insert(0, Path(path).stem)
                
    def _load_model(self):
        """Load selected character model."""
        import lunavox_tts as lunavox
        from lunavox_tts import unload_character
        from main import REPO_ROOT
        
        char = self.char_cb.get()
        if not char:
            return
            
        version = self.version_cb.get()
        char_dir = REPO_ROOT / "CharacterData" / "model" / version / char
        
        self.app.set_status(get_text(self.app.lang, "status_loading"), show_progress=True)
        self.load_indicator.configure(text="⌛ " + get_text(self.app.lang, "status_loading"), foreground=COLORS["warning"])
        self.btn_load.configure(state="disabled")

        def task():
            try:
                if self.app.loaded_character:
                    unload_character(self.app.loaded_character)
                    
                lunavox.load_character(char, str(char_dir))
                self.app.update_model_status(character=char)
                self.app.set_status(get_text(self.app.lang, "model_loaded"))
                self.load_indicator.configure(text="✅ " + get_text(self.app.lang, "model_loaded"), foreground=COLORS["success"])
            except Exception as e:
                self.app.set_status(get_text(self.app.lang, "status_error", str(e)))
                self.load_indicator.configure(text="❌ " + get_text(self.app.lang, "error"), foreground=COLORS["error"])
                messagebox.showerror(get_text(self.app.lang, "error"), str(e))
            finally:
                self.btn_load.configure(state="normal")
        
        threading.Thread(target=task, daemon=True).start()
        
    def _unload_model(self):
        """Unload current character model."""
        from lunavox_tts import unload_character
        
        if self.app.loaded_character:
            unload_character(self.app.loaded_character)
            self.app.update_model_status()
            self.app.set_status(get_text(self.app.lang, "status_ready"))
            self.load_indicator.configure(text="")
            # Reset persona and ref statuses
            self._reset_persona_status()
            self._reset_ref_status()
            
    def _load_persona(self):
        """Load selected persona."""
        import lunavox_tts as lunavox
        
        if not self.app.loaded_character:
            messagebox.showwarning(
                get_text(self.app.lang, "warning"),
                get_text(self.app.lang, "no_model")
            )
            return
            
        persona_name = self.persona_cb.get()
        persona_path = self.persona_paths.get(persona_name)
        
        if not persona_path:
            return
            
        self.app.set_status(get_text(self.app.lang, "status_loading"), show_progress=True)
        self.persona_status.set_loading()
        self.btn_load_persona.configure(state="disabled")
        
        def task():
            try:
                lunavox.load_persona(self.app.loaded_character, persona_path)
                self.app.update_model_status(
                    character=self.app.loaded_character,
                    persona=persona_name
                )
                self.app.set_status(get_text(self.app.lang, "persona_loaded"))
                self.after(0, lambda: self.persona_status.set_success(
                    get_text(self.app.lang, "persona_loaded")
                ))
            except Exception as e:
                self.app.set_status(get_text(self.app.lang, "status_error", str(e)))
                self.after(0, lambda: self.persona_status.set_error(
                    get_text(self.app.lang, "error")
                ))
                messagebox.showerror(get_text(self.app.lang, "error"), str(e))
            finally:
                self.btn_load_persona.configure(state="normal")
                
        threading.Thread(target=task, daemon=True).start()
    
    def _on_persona_selected(self, event=None):
        """Handle persona selection change - update language display."""
        persona_name = self.persona_cb.get()
        persona_path = self.persona_paths.get(persona_name)
        
        if persona_path:
            lang = self._get_persona_language(persona_path)
            if lang:
                lang_key = LANG_KEY_MAP.get(lang, "lang_auto")
                localized_lang = get_text(self.app.lang, lang_key)
                self.persona_lang_label.configure(text=f"🌐 {localized_lang}")
            else:
                self.persona_lang_label.configure(text="")
    
    def _get_persona_language(self, persona_path: str) -> str:
        """Read language from persona metadata.json."""
        try:
            metadata_path = os.path.join(persona_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata.get("language", "")
        except Exception:
            pass
        return ""
    
    def _reset_persona_status(self):
        """Reset persona status to 'not loaded'."""
        self.persona_status.set_status(
            get_text(self.app.lang, "no_model").replace("Model", "Persona"),
            "○",
            COLORS["text_muted"]
        )
    
    def _reset_ref_status(self):
        """Reset reference audio status to 'not set'."""
        self.ref_status.set_status(
            get_text(self.app.lang, "no_model").replace("Model", "Ref"),
            "○",
            COLORS["text_muted"]
        )
        
    def _synthesize(self):
        """Perform TTS synthesis."""
        import lunavox_tts as lunavox
        from main import REPO_ROOT
        
        char = self.app.loaded_character
        if not char:
            messagebox.showwarning(
                get_text(self.app.lang, "warning"),
                get_text(self.app.lang, "no_model")
            )
            return
            
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            return
            
        target_lang = self.target_lang_cb.get()
        mode = self.tts_mode.get()
        
        self.app.set_status(get_text(self.app.lang, "status_synthesizing"), show_progress=True)
        self.btn_synth.configure(state="disabled")
        self.latency_display.clear()
        
        def task():
            try:
                # If in reference mode and no persona loaded, set reference audio
                if mode == "reference":
                    ref_audio = self.ref_audio_entry.get()
                    ref_text = self.ref_text_entry.get()
                    ref_lang = self.ref_lang_cb.get()
                    
                    if ref_audio and ref_text:
                        lunavox.set_reference_audio(char, ref_audio, ref_text, audio_language=ref_lang)
                        self.after(0, lambda: self.ref_status.set_success(
                            get_text(self.app.lang, "status_success")
                        ))

                # Determine output path based on auto-save setting
                if self.auto_save.get():
                    output_path = REPO_ROOT / "Output" / f"gui_out_{int(time.time())}.wav"
                    output_path.parent.mkdir(exist_ok=True)
                    is_temp = False
                else:
                    # Use temp file - will be discarded unless saved
                    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="lunavox_")
                    os.close(temp_fd)
                    output_path = Path(temp_path)
                    is_temp = True
                    # Clean up previous temp file if exists
                    self._cleanup_temp_audio()
                
                # Measure synthesis time
                start_time = time.perf_counter()
                
                lunavox.tts(
                    character_name=char,
                    text=text,
                    play=True,
                    language=target_lang,
                    save_path=str(output_path)
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                # Get audio duration
                audio_duration_ms = self._get_audio_duration(str(output_path))
                
                # Update UI on main thread
                self.after(0, lambda: self._on_synthesis_complete(
                    str(output_path), latency_ms, audio_duration_ms, is_temp
                ))
                
            except Exception as e:
                self.app.set_status(get_text(self.app.lang, "status_error", str(e)))
                messagebox.showerror(get_text(self.app.lang, "error"), str(e))
            finally:
                self.btn_synth.configure(state="normal")
                
        threading.Thread(target=task, daemon=True).start()
    
    def _on_synthesis_complete(self, output_path: str, latency_ms: float, audio_duration_ms: float, is_temp: bool = False):
        """Handle synthesis completion with metrics."""
        self.last_output_path = output_path
        self.last_audio_duration = audio_duration_ms / 1000 if audio_duration_ms else None
        
        # Track temp file for cleanup
        if is_temp:
            self.temp_audio_path = output_path
        else:
            self.temp_audio_path = None
        
        # Update latency display
        self.latency_display.update(latency_ms, audio_duration_ms)
        
        # Update audio panel with temp indicator
        if is_temp:
            self.audio_panel.set_audio(output_path, self.last_audio_duration)
            self.audio_panel.file_info_var.set(
                self.audio_panel.file_info_var.get() + " [Temp - Save to keep]"
            )
        else:
            self.audio_panel.set_audio(output_path, self.last_audio_duration)
        
        self.app.set_status(get_text(self.app.lang, "status_success"))
    
    def _cleanup_temp_audio(self):
        """Clean up temporary audio file."""
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                os.remove(self.temp_audio_path)
            except Exception:
                pass
            self.temp_audio_path = None

    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in milliseconds."""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return (frames / rate) * 1000
        except Exception:
            return None
    
    def _save_audio(self, source_path: str):
        """Save audio file to user-selected location."""
        if not source_path or not os.path.exists(source_path):
            return
            
        dest_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=os.path.basename(source_path)
        )
        
        if dest_path:
            try:
                shutil.copy2(source_path, dest_path)
                messagebox.showinfo(
                    get_text(self.app.lang, "success"),
                    f"Audio saved to:\n{dest_path}"
                )
            except Exception as e:
                messagebox.showerror(get_text(self.app.lang, "error"), str(e))
    
    def _play_audio(self, audio_path: str):
        """Play audio file."""
        import lunavox_tts as lunavox
        try:
            from lunavox_tts.Audio.AudioPlayer import AudioPlayer
            player = AudioPlayer()
            player.play_file(audio_path)
        except Exception:
            # Fallback to system player
            import subprocess
            import sys
            if sys.platform == 'darwin':
                subprocess.run(['afplay', audio_path])
            elif sys.platform == 'win32':
                os.startfile(audio_path)
            else:
                subprocess.run(['aplay', audio_path])
        
    def _stop(self):
        """Stop current TTS playback."""
        import lunavox_tts as lunavox
        lunavox.stop()
        self.app.set_status(get_text(self.app.lang, "status_ready"))
        
    def update_ui_texts(self):
        """Update UI text when language changes."""
        lang = self.app.lang
        
        # Update labels and buttons
        self.radio_persona.configure(text=get_text(lang, "tts_mode_persona"))
        self.radio_reference.configure(text=get_text(lang, "tts_mode_reference"))
        self.btn_load.configure(text=get_text(lang, "load_model"))
        self.btn_unload.configure(text=get_text(lang, "unload_model"))
        self.btn_load_persona.configure(text=get_text(lang, "load_persona"))
        self.btn_synth.configure(text=get_text(lang, "synthesize"))
        self.btn_stop.configure(text=get_text(lang, "stop"))
