
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import sys

# Constants and I18N
from i18n import I18N

class InferenceTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.build_ui()
        
    def build_ui(self):
        # Configuration Section
        cfg_frame = ttk.LabelFrame(self, text=I18N[self.app.lang]["character"], padding=10)
        cfg_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(cfg_frame, text=I18N[self.app.lang]["version"]).grid(row=0, column=0, sticky="w", padx=5)
        self.version_cb = ttk.Combobox(cfg_frame, values=["v2", "v2_pro_plus"], state="readonly")
        self.version_cb.set("v2")
        self.version_cb.grid(row=0, column=1, sticky="ew", padx=5)
        self.version_cb.bind("<<ComboboxSelected>>", lambda e: self.on_version_change())
        
        ttk.Label(cfg_frame, text=I18N[self.app.lang]["character"]).grid(row=0, column=2, sticky="w", padx=5)
        self.char_cb = ttk.Combobox(cfg_frame, state="readonly")
        self.char_cb.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        
        btn_frame = ttk.Frame(cfg_frame)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        self.btn_load = ttk.Button(btn_frame, text=I18N[self.app.lang]["load_model"], command=self.load_model)
        self.btn_load.pack(side="left", padx=5)
        
        self.btn_unload = ttk.Button(btn_frame, text=I18N[self.app.lang]["unload_model"], command=self.unload_model)
        self.btn_unload.pack(side="left", padx=5)

        # Reference Section
        ref_frame = ttk.LabelFrame(self, text=I18N[self.app.lang]["ref_audio"], padding=10)
        ref_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(ref_frame, text=I18N[self.app.lang]["preset_audio"]).grid(row=0, column=0, sticky="w", padx=5)
        self.preset_cb = ttk.Combobox(ref_frame, state="readonly")
        self.preset_cb.grid(row=0, column=1, sticky="ew", padx=5)
        self.preset_cb.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        ttk.Label(ref_frame, text=I18N[self.app.lang]["ref_lang"]).grid(row=0, column=2, sticky="w", padx=5)
        self.ref_lang_cb = ttk.Combobox(ref_frame, values=["en", "zh", "ja"], state="readonly")
        self.ref_lang_cb.set("ja")
        self.ref_lang_cb.grid(row=0, column=3, sticky="ew", padx=5)
        self.ref_lang_cb.bind("<<ComboboxSelected>>", lambda e: self.update_preset_audio_list())
        
        ttk.Label(ref_frame, text=I18N[self.app.lang]["ref_audio"]).grid(row=1, column=0, sticky="w", padx=5)
        self.ref_audio_entry = ttk.Entry(ref_frame)
        self.ref_audio_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(ref_frame, text=I18N[self.app.lang]["browse"], command=self.browse_ref_audio).grid(row=1, column=2, sticky="w")
        
        ttk.Label(ref_frame, text=I18N[self.app.lang]["ref_text"]).grid(row=2, column=0, sticky="w", padx=5)
        self.ref_text_entry = ttk.Entry(ref_frame)
        self.ref_text_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        # Synthesis Section
        synth_frame = ttk.LabelFrame(self, text=I18N[self.app.lang]["tab_inference"], padding=10)
        synth_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Label(synth_frame, text=I18N[self.app.lang]["target_lang"]).pack(anchor="w", padx=5)
        self.target_lang_cb = ttk.Combobox(synth_frame, values=["en", "zh", "ja"], state="readonly")
        self.target_lang_cb.set("zh")
        self.target_lang_cb.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(synth_frame, text=I18N[self.app.lang]["target_text"]).pack(anchor="w", padx=5)
        self.text_input = tk.Text(synth_frame, height=8, font=("Segoe UI", 10))
        self.text_input.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.btn_synth = ttk.Button(synth_frame, text=I18N[self.app.lang]["synthesize"], style="Primary.TButton", command=self.synthesize)
        self.btn_synth.pack(pady=10)
        
        # Initial scan
        self.update_character_list()
        self.update_preset_audio_list()

    def update_ui_texts(self):
        # Update labels and button texts when language changes
        # Recursive update is overkill, but for a simple app we update key ones:
        lang = self.app.lang
        self.btn_load.configure(text=I18N[lang]["load_model"])
        self.btn_unload.configure(text=I18N[lang]["unload_model"])
        self.btn_synth.configure(text=I18N[lang]["synthesize"])
        # (Other labels would need references to be updated cleanly)

    def on_version_change(self):
        self.update_character_list()
        
    def update_character_list(self):
        from main import REPO_ROOT
        v = self.version_cb.get()
        base_dir = REPO_ROOT / "CharacterData" / "character_model" / v
        if not base_dir.exists():
            self.char_cb['values'] = []
            return
        chars = [p.name for p in base_dir.iterdir() if p.is_dir()]
        self.char_cb['values'] = sorted(chars)
        if chars: self.char_cb.set(chars[0])

    def update_preset_audio_list(self):
        from main import REPO_ROOT
        lang_map = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
        lang_folder = lang_map.get(self.ref_lang_cb.get(), "Japanese")
        audio_dir = REPO_ROOT / "CharacterData" / "audio_resources" / lang_folder
        if not audio_dir.exists():
            self.preset_cb['values'] = []
            return
        wavs = list(audio_dir.glob("*.wav"))
        self.preset_cb['values'] = [w.name for w in wavs]
        self.preset_audio_paths = {w.name: str(w) for w in wavs}

    def on_preset_selected(self, event):
        filename = self.preset_cb.get()
        path = self.preset_audio_paths.get(filename)
        if path:
            self.ref_audio_entry.delete(0, tk.END)
            self.ref_audio_entry.insert(0, path)
            self.ref_text_entry.delete(0, tk.END)
            self.ref_text_entry.insert(0, Path(path).stem)

    def browse_ref_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.ref_audio_entry.delete(0, tk.END)
            self.ref_audio_entry.insert(0, path)
            if not self.ref_text_entry.get():
                self.ref_text_entry.insert(0, Path(path).stem)

    def load_model(self):
        import lunavox_tts as lunavox
        from lunavox_tts import unload_character
        from main import REPO_ROOT
        
        char = self.char_cb.get()
        if not char: return
        v = self.version_cb.get()
        char_dir = REPO_ROOT / "CharacterData" / "character_model" / v / char
        
        self.app.set_status(I18N[self.app.lang]["status_loading"])
        
        def task():
            try:
                if self.app.loaded_character:
                    unload_character(self.app.loaded_character)
                lunavox.load_character(char, str(char_dir))
                self.app.loaded_character = char
                self.app.set_status(f"Loaded: {char}")
            except Exception as e:
                self.app.set_status(I18N[self.app.lang]["status_error"].format(str(e)))
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=task).start()

    def unload_model(self):
        from lunavox_tts import unload_character
        if self.app.loaded_character:
            unload_character(self.app.loaded_character)
            self.app.set_status(f"Unloaded: {self.app.loaded_character}")
            self.app.loaded_character = None
        else:
            self.app.set_status("No character loaded")

    def synthesize(self):
        import lunavox_tts as lunavox
        from main import REPO_ROOT
        char = self.app.loaded_character
        if not char:
            messagebox.showwarning("Warning", "Please load a character first")
            return
        
        text = self.text_input.get("1.0", tk.END).strip()
        if not text: return
        
        ref_audio = self.ref_audio_entry.get()
        ref_text = self.ref_text_entry.get()
        ref_lang = self.ref_lang_cb.get()
        target_lang = self.target_lang_cb.get()
        
        self.app.set_status("Synthesizing...")
        self.btn_synth.configure(state="disabled")
        
        def task():
            try:
                if ref_audio and ref_text:
                    lunavox.set_reference_audio(char, ref_audio, ref_text, audio_language=ref_lang)
                
                output_path = REPO_ROOT / "Output" / f"gui_out_{int(time.time())}.wav"
                output_path.parent.mkdir(exist_ok=True)
                
                lunavox.tts(
                    character_name=char,
                    text=text,
                    play=True,
                    language=target_lang,
                    save_path=str(output_path)
                )
                self.app.set_status(I18N[self.app.lang]["status_success"])
            except Exception as e:
                self.app.set_status(I18N[self.app.lang]["status_error"].format(str(e)))
                messagebox.showerror("Error", str(e))
            finally:
                self.btn_synth.configure(state="normal")
        
        threading.Thread(target=task).start()
