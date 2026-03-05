"""
LunaVox Persona Creation Tab
Create and manage personas from reference audio files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import os

from i18n import get_text
from theme import COLORS, FONTS, SPACING, get_text_widget_config


class PersonaTab(ttk.Frame):
    """Tab for creating personas from reference audio."""
    
    def __init__(self, parent, app):
        super().__init__(parent, padding=SPACING["lg"])
        self.app = app
        self.configure(style="TFrame")
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the persona creation UI."""
        
        # Header
        header = ttk.Label(
            self,
            text="✨ " + get_text(self.app.lang, "tab_persona").replace("✨ ", ""),
            style="Header.TLabel"
        )
        header.pack(anchor="w", pady=(0, SPACING["lg"]))
        
        # Info text
        info_text = ttk.Label(
            self,
            text="Create a saved voice profile from reference audio for quick TTS without re-processing.",
            style="Muted.TLabel",
            wraplength=700
        )
        info_text.pack(anchor="w", pady=(0, SPACING["lg"]))
        
        # Main form
        form_frame = ttk.LabelFrame(
            self,
            text=get_text(self.app.lang, "create_persona"),
            padding=SPACING["lg"]
        )
        form_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        # Persona name
        self._add_form_row(form_frame, "persona_name", 0)
        self.name_entry = ttk.Entry(form_frame, width=40)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=SPACING["sm"], pady=SPACING["xs"])
        
        # Source audio
        self._add_form_row(form_frame, "persona_audio", 1)
        audio_frame = ttk.Frame(form_frame, style="Card.TFrame")
        audio_frame.grid(row=1, column=1, sticky="ew", padx=SPACING["sm"], pady=SPACING["xs"])
        
        self.audio_entry = ttk.Entry(audio_frame)
        self.audio_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(audio_frame, text=get_text(self.app.lang, "browse"), command=self._browse_audio).pack(side="right", padx=(SPACING["xs"], 0))
        
        # Audio text
        self._add_form_row(form_frame, "persona_text", 2)
        self.text_entry = ttk.Entry(form_frame, width=40)
        self.text_entry.grid(row=2, column=1, sticky="ew", padx=SPACING["sm"], pady=SPACING["xs"])
        
        # Language
        self._add_form_row(form_frame, "persona_lang", 3)
        self.lang_cb = ttk.Combobox(form_frame, values=["auto", "en", "zh", "ja"], state="readonly", width=15)
        self.lang_cb.set("auto")
        self.lang_cb.grid(row=3, column=1, sticky="w", padx=SPACING["sm"], pady=SPACING["xs"])
        
        # Model version
        self._add_form_row(form_frame, "version", 4)
        self.version_cb = ttk.Combobox(form_frame, values=["v2", "v2_pro_plus"], state="readonly", width=15)
        self.version_cb.set("v2")
        self.version_cb.grid(row=4, column=1, sticky="w", padx=SPACING["sm"], pady=SPACING["xs"])
        
        form_frame.columnconfigure(1, weight=1)
        
        # Create button
        btn_frame = ttk.Frame(self, style="TFrame")
        btn_frame.pack(fill="x", pady=SPACING["lg"])
        
        self.btn_create = ttk.Button(
            btn_frame,
            text=get_text(self.app.lang, "create_persona"),
            command=self._create_persona,
            style="Primary.TButton"
        )
        self.btn_create.pack(side="left")
        
        # Progress section
        self.progress_frame = ttk.Frame(self, style="TFrame")
        
        self.progress_label = ttk.Label(
            self.progress_frame,
            text="",
            style="TLabel"
        )
        self.progress_label.pack(anchor="w")
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode="indeterminate",
            length=400
        )
        self.progress_bar.pack(fill="x", pady=SPACING["sm"])
        
        # Existing personas list
        self._build_persona_list()
        
    def _add_form_row(self, parent, label_key, row):
        """Add a form row with label."""
        label = ttk.Label(
            parent,
            text=get_text(self.app.lang, label_key),
            style="Card.TLabel"
        )
        label.grid(row=row, column=0, sticky="w", padx=SPACING["sm"], pady=SPACING["xs"])
        
    def _build_persona_list(self):
        """Build the existing personas list section."""
        list_frame = ttk.LabelFrame(
            self,
            text=get_text(self.app.lang, "select_persona"),
            padding=SPACING["md"]
        )
        list_frame.pack(fill="both", expand=True)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame, style="Card.TFrame")
        list_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container, orient="vertical")
        
        self.persona_listbox = tk.Listbox(
            list_container,
            bg=COLORS["bg_input"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["primary"],
            selectforeground=COLORS["text_inverse"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border_default"],
            highlightcolor=COLORS["border_focus"],
            font=(FONTS["family"], FONTS["size_base"]),
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.persona_listbox.yview)
        
        self.persona_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Refresh button
        ttk.Button(
            list_frame,
            text="↻ Refresh",
            command=self._refresh_persona_list
        ).pack(anchor="e", pady=(SPACING["sm"], 0))
        
        # Initial load
        self._refresh_persona_list()
        
    def _browse_audio(self):
        """Browse for audio file."""
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.flac *.ogg"), ("All files", "*.*")]
        )
        if path:
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, path)
            
            # Auto-fill name and text from filename
            stem = Path(path).stem
            if not self.name_entry.get():
                self.name_entry.insert(0, stem)
            if not self.text_entry.get():
                self.text_entry.insert(0, stem)
                
    def _create_persona(self):
        """Create persona from the form data."""
        name = self.name_entry.get().strip()
        audio_path = self.audio_entry.get().strip()
        text = self.text_entry.get().strip()
        lang = self.lang_cb.get()
        version = self.version_cb.get()
        
        # Validation
        if not name:
            messagebox.showwarning(
                get_text(self.app.lang, "warning"),
                "Please enter a persona name."
            )
            return
            
        if not audio_path or not Path(audio_path).exists():
            messagebox.showwarning(
                get_text(self.app.lang, "warning"),
                "Please select a valid audio file."
            )
            return
            
        if not text:
            text = Path(audio_path).stem
            
        # Show progress
        self.progress_frame.pack(fill="x", pady=SPACING["md"])
        self.progress_label.configure(text=get_text(self.app.lang, "persona_creating"))
        self.progress_bar.start(10)
        self.btn_create.configure(state="disabled")
        
        def task():
            try:
                from main import REPO_ROOT
                import lunavox_tts as lunavox
                
                # Output directory
                output_dir = REPO_ROOT / "lunavoxData" / "CharacterData" / "character" / name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create persona using lunavox API
                lunavox.create_persona(
                    character_name=name,
                    audio_path=audio_path,
                    audio_text=text,
                    save_dir=str(output_dir),
                    audio_language=lang if lang != "auto" else None
                )
                
                # Success
                self.after(0, lambda: self._on_persona_created(name))
                
            except Exception as e:
                self.after(0, lambda: self._on_persona_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()
        
    def _on_persona_created(self, name):
        """Handle successful persona creation."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.btn_create.configure(state="normal")
        
        # Clear form
        self.name_entry.delete(0, tk.END)
        self.audio_entry.delete(0, tk.END)
        self.text_entry.delete(0, tk.END)
        
        # Refresh list
        self._refresh_persona_list()
        
        # Show success
        messagebox.showinfo(
            get_text(self.app.lang, "success"),
            get_text(self.app.lang, "persona_success")
        )
        
        self.app.set_status(get_text(self.app.lang, "persona_success"))
        
        # Also refresh inference tab's persona list
        if hasattr(self.app, 'inf_tab'):
            self.app.inf_tab._update_persona_list()
            
    def _on_persona_error(self, error_msg):
        """Handle persona creation error."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.btn_create.configure(state="normal")
        
        messagebox.showerror(
            get_text(self.app.lang, "error"),
            error_msg
        )
        
        self.app.set_status(get_text(self.app.lang, "status_error", error_msg))
        
    def _refresh_persona_list(self):
        """Refresh the persona listbox."""
        from main import REPO_ROOT
        
        self.persona_listbox.delete(0, tk.END)
        
        persona_dir = REPO_ROOT / "lunavoxData" / "CharacterData" / "character"
        if not persona_dir.exists():
            return
            
        for p in sorted(persona_dir.iterdir()):
            if p.is_dir():
                # Check if valid persona
                if (p / "features.npz").exists() or (p / "metadata.json").exists():
                    self.persona_listbox.insert(tk.END, f"  ✓  {p.name}")
                else:
                    self.persona_listbox.insert(tk.END, f"  ?  {p.name}")
                    
    def update_ui_texts(self):
        """Update UI text when language changes."""
        lang = self.app.lang
        self.btn_create.configure(text=get_text(lang, "create_persona"))
