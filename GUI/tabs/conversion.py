"""
LunaVox Model Conversion Tab
Convert PyTorch models to ONNX format.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from i18n import get_text
from theme import COLORS, FONTS, SPACING


class ConversionTab(ttk.Frame):
    """Tab for converting PyTorch models to ONNX format."""
    
    def __init__(self, parent, app):
        super().__init__(parent, padding=SPACING["lg"])
        self.app = app
        self.configure(style="TFrame")
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the conversion tab UI."""
        
        # Header
        header = ttk.Label(
            self,
            text="🔄 " + get_text(self.app.lang, "tab_conversion").replace("🔄 ", ""),
            style="Header.TLabel"
        )
        header.pack(anchor="w", pady=(0, SPACING["md"]))
        
        # Info text
        info_text = ttk.Label(
            self,
            text="Convert GPT-SoVITS PyTorch checkpoints (.ckpt, .pth) to ONNX format for inference.",
            style="Muted.TLabel",
            wraplength=700
        )
        info_text.pack(anchor="w", pady=(0, SPACING["lg"]))
        
        # Main form
        form_frame = ttk.LabelFrame(
            self,
            text=get_text(self.app.lang, "convert"),
            padding=SPACING["lg"]
        )
        form_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        # Source path
        row1 = ttk.Frame(form_frame, style="Card.TFrame")
        row1.pack(fill="x", pady=SPACING["sm"])
        
        ttk.Label(row1, text=get_text(self.app.lang, "source_path"), style="Card.TLabel").pack(anchor="w")
        ttk.Label(row1, text=get_text(self.app.lang, "source_path_hint"), style="Muted.TLabel").pack(anchor="w")
        
        src_frame = ttk.Frame(row1, style="Card.TFrame")
        src_frame.pack(fill="x", pady=(SPACING["xs"], 0))
        
        self.src_entry = ttk.Entry(src_frame)
        self.src_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(src_frame, text=get_text(self.app.lang, "browse"), command=self._browse_src).pack(side="right", padx=(SPACING["xs"], 0))
        
        # Model name
        row2 = ttk.Frame(form_frame, style="Card.TFrame")
        row2.pack(fill="x", pady=SPACING["sm"])
        
        ttk.Label(row2, text=get_text(self.app.lang, "char_name"), style="Card.TLabel").pack(anchor="w")
        self.name_entry = ttk.Entry(row2, width=40)
        self.name_entry.pack(anchor="w", pady=(SPACING["xs"], 0))
        
        # Model version (auto-detect or override)
        row3 = ttk.Frame(form_frame, style="Card.TFrame")
        row3.pack(fill="x", pady=SPACING["sm"])
        
        ttk.Label(row3, text=get_text(self.app.lang, "version") + " (auto-detect)", style="Card.TLabel").pack(side="left")
        self.version_cb = ttk.Combobox(row3, values=["auto", "v2", "v2Pro", "v2ProPlus"], state="readonly", width=15)
        self.version_cb.set("auto")
        self.version_cb.pack(side="left", padx=SPACING["sm"])
        
        # Output directory
        row4 = ttk.Frame(form_frame, style="Card.TFrame")
        row4.pack(fill="x", pady=SPACING["sm"])
        
        ttk.Label(row4, text=get_text(self.app.lang, "output_root"), style="Card.TLabel").pack(anchor="w")
        
        out_frame = ttk.Frame(row4, style="Card.TFrame")
        out_frame.pack(fill="x", pady=(SPACING["xs"], 0))
        
        self.out_entry = ttk.Entry(out_frame)
        self.out_entry.pack(side="left", fill="x", expand=True)
        
        # Set default output path
        from main import REPO_ROOT
        default_out = REPO_ROOT / "lunavoxData" / "CharacterData" / "model" / "v2"
        self.out_entry.insert(0, str(default_out))
        
        ttk.Button(out_frame, text=get_text(self.app.lang, "browse"), command=self._browse_out).pack(side="right", padx=(SPACING["xs"], 0))
        
        # Convert button
        btn_frame = ttk.Frame(self, style="TFrame")
        btn_frame.pack(fill="x", pady=SPACING["md"])
        
        self.btn_convert = ttk.Button(
            btn_frame,
            text=get_text(self.app.lang, "convert"),
            command=self._convert,
            style="Primary.TButton"
        )
        self.btn_convert.pack(side="left")
        
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
        
        # Log output
        log_frame = ttk.LabelFrame(
            self,
            text="Conversion Log",
            padding=SPACING["md"]
        )
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(
            log_frame,
            height=10,
            bg=COLORS["bg_input"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border_default"],
            font=(FONTS["family_mono"], FONTS["size_sm"]),
            state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)
        
    def _browse_src(self):
        """Browse for source directory."""
        path = filedialog.askdirectory()
        if path:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, path)
            
            # Try to auto-fill name from directory name
            if not self.name_entry.get():
                from pathlib import Path
                self.name_entry.insert(0, Path(path).name)
                
    def _browse_out(self):
        """Browse for output directory."""
        path = filedialog.askdirectory()
        if path:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, path)
            
    def _log(self, message: str):
        """Add message to log output."""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        
    def _convert(self):
        """Start model conversion."""
        src = self.src_entry.get().strip()
        name = self.name_entry.get().strip()
        out_root = self.out_entry.get().strip()
        version = self.version_cb.get()
        
        # Validation
        if not src or not name or not out_root:
            messagebox.showwarning(
                get_text(self.app.lang, "warning"),
                "All fields are required."
            )
            return
            
        # Show progress
        self.progress_frame.pack(fill="x", before=self.log_text.master)
        self.progress_label.configure(text=get_text(self.app.lang, "converting"))
        self.progress_bar.start(10)
        self.btn_convert.configure(state="disabled")
        
        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")
        
        def task():
            try:
                from lunavox_tts.Converter.v2.Converter import find_ckpt_and_pth
                from converter import convert
                
                self.after(0, lambda: self._log(f"🔍 Scanning source: {src}"))
                
                ckpt, pth = find_ckpt_and_pth(src)
                if not (ckpt and pth):
                    raise FileNotFoundError("Could not find .ckpt and .pth files in source directory")
                    
                self.after(0, lambda: self._log(f"✓ Found checkpoint: {ckpt}"))
                self.after(0, lambda: self._log(f"✓ Found weights: {pth}"))
                
                dest_dir = os.path.join(out_root, name)
                self.after(0, lambda: self._log(f"\n🔄 Converting to: {dest_dir}"))
                
                # Perform conversion
                version_arg = None if version == "auto" else version
                convert(
                    ckpt_path=ckpt,
                    pth_path=pth,
                    output_dir=dest_dir,
                    model_version=version_arg
                )
                
                self.after(0, lambda: self._on_convert_success(name, dest_dir))
                
            except Exception as e:
                self.after(0, lambda: self._on_convert_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()
        
    def _on_convert_success(self, name, dest_dir):
        """Handle successful conversion."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.btn_convert.configure(state="normal")
        
        self._log(f"\n✅ {get_text(self.app.lang, 'conversion_success')}")
        self._log(f"   Output: {dest_dir}")
        
        messagebox.showinfo(
            get_text(self.app.lang, "success"),
            f"Model converted successfully!\n\nOutput: {dest_dir}"
        )
        
        self.app.set_status(get_text(self.app.lang, "conversion_success"))
        
        # Refresh inference tab's character list
        if hasattr(self.app, 'inf_tab'):
            self.app.inf_tab._update_character_list()
            
    def _on_convert_error(self, error_msg):
        """Handle conversion error."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.btn_convert.configure(state="normal")
        
        self._log(f"\n❌ Error: {error_msg}")
        
        messagebox.showerror(
            get_text(self.app.lang, "error"),
            error_msg
        )
        
        self.app.set_status(get_text(self.app.lang, "status_error", error_msg))
        
    def update_ui_texts(self):
        """Update UI text when language changes."""
        lang = self.app.lang
        self.btn_convert.configure(text=get_text(lang, "convert"))
