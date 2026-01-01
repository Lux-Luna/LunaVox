"""
LunaVox Settings Tab
Application settings and preferences.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import os
import threading

from i18n import get_text, I18N
from theme import COLORS, FONTS, SPACING, apply_theme, get_current_theme

# Version constant
LUNAVOX_VERSION = "1.5.0"


class SettingsTab(ttk.Frame):
    """Tab for application settings."""
    
    def __init__(self, parent, app):
        super().__init__(parent, padding=SPACING["lg"])
        self.app = app
        self.configure(style="TFrame")
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the settings tab UI."""
        
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
        
        # Header
        header = ttk.Label(
            self.scrollable_frame,
            text="⚙ " + get_text(self.app.lang, "tab_settings").replace("⚙ ", ""),
            style="Header.TLabel"
        )
        header.pack(anchor="w", pady=(0, SPACING["lg"]))
        
        # Appearance section
        self._build_appearance_section()
        
        # Language settings
        self._build_language_section()
        
        # Runtime settings (CPU/GPU)
        self._build_runtime_section()
        
        # Cache management
        self._build_cache_section()
        
        # About section
        self._build_about_section()
        
    def _build_appearance_section(self):
        """Build the appearance/theme section."""
        appearance_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="🎨 Appearance",
            padding=SPACING["lg"]
        )
        appearance_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        theme_row = ttk.Frame(appearance_frame, style="Card.TFrame")
        theme_row.pack(fill="x")
        
        ttk.Label(theme_row, text="Theme:", style="Card.TLabel").pack(side="left")
        
        self.theme_var = tk.StringVar(value=get_current_theme())
        
        themes = [
            ("🌙 Dark", "dark"),
            ("☀️ Light", "light")
        ]
        
        for display_name, theme_name in themes:
            rb = ttk.Radiobutton(
                theme_row,
                text=display_name,
                variable=self.theme_var,
                value=theme_name,
                command=self._on_theme_change,
                style="Card.TRadiobutton"
            )
            rb.pack(side="left", padx=SPACING["lg"])
            
    def _build_language_section(self):
        """Build the language settings section."""
        lang_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "ui_language"),
            padding=SPACING["lg"]
        )
        lang_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        lang_row = ttk.Frame(lang_frame, style="Card.TFrame")
        lang_row.pack(fill="x")
        
        # Language buttons
        self.lang_var = tk.StringVar(value=self.app.lang)
        
        languages = [
            ("🇺🇸 English", "en"),
            ("🇨🇳 中文", "zh"),
            ("🇯🇵 日本語", "ja")
        ]
        
        for display_name, lang_code in languages:
            rb = ttk.Radiobutton(
                lang_row,
                text=display_name,
                variable=self.lang_var,
                value=lang_code,
                command=self._on_language_change,
                style="Card.TRadiobutton"
            )
            rb.pack(side="left", padx=SPACING["lg"])
            
    def _build_runtime_section(self):
        """Build the CPU/GPU runtime section."""
        runtime_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="⚙ Runtime",
            padding=SPACING["lg"]
        )
        runtime_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        # Current runtime status
        status_row = ttk.Frame(runtime_frame, style="Card.TFrame")
        status_row.pack(fill="x", pady=(0, SPACING["sm"]))
        
        ttk.Label(status_row, text="Current:", style="Card.TLabel").pack(side="left")
        
        self.runtime_status_label = ttk.Label(
            status_row,
            text="Checking...",
            style="Card.TLabel"
        )
        self.runtime_status_label.pack(side="left", padx=SPACING["sm"])
        
        # Note about restart
        note_label = ttk.Label(
            runtime_frame,
            text="⚠️ Changing runtime requires application restart",
            style="Muted.TLabel"
        )
        note_label.pack(anchor="w", pady=(0, SPACING["sm"]))
        
        # Runtime selection
        runtime_row = ttk.Frame(runtime_frame, style="Card.TFrame")
        runtime_row.pack(fill="x")
        
        self.runtime_var = tk.StringVar(value="cpu")
        
        self.rb_cpu = ttk.Radiobutton(
            runtime_row,
            text="🖥️ CPU (Universal)",
            variable=self.runtime_var,
            value="cpu",
            style="Card.TRadiobutton"
        )
        self.rb_cpu.pack(side="left", padx=(0, SPACING["lg"]))
        
        self.rb_gpu = ttk.Radiobutton(
            runtime_row,
            text="🎮 GPU (CUDA)",
            variable=self.runtime_var,
            value="gpu",
            style="Card.TRadiobutton"
        )
        self.rb_gpu.pack(side="left", padx=(0, SPACING["lg"]))
        
        self.btn_apply_runtime = ttk.Button(
            runtime_row,
            text="Apply",
            command=self._apply_runtime_change
        )
        self.btn_apply_runtime.pack(side="left")
        
        # Load current runtime
        self._update_runtime_status()
        
    def _build_cache_section(self):
        """Build the cache management section."""
        cache_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "clean_cache"),
            padding=SPACING["lg"]
        )
        cache_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        cache_info = ttk.Frame(cache_frame, style="Card.TFrame")
        cache_info.pack(fill="x", pady=(0, SPACING["sm"]))
        
        self.cache_size_label = ttk.Label(
            cache_info,
            text="Cache size: calculating...",
            style="Card.TLabel"
        )
        self.cache_size_label.pack(side="left")
        
        self.btn_clean_cache = ttk.Button(
            cache_frame,
            text=get_text(self.app.lang, "clean_cache"),
            command=self._clean_cache
        )
        self.btn_clean_cache.pack(anchor="w")
        
        # Update cache size
        self._update_cache_size()
        
    def _build_about_section(self):
        """Build the about section."""
        about_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=get_text(self.app.lang, "about"),
            padding=SPACING["lg"]
        )
        about_frame.pack(fill="x", pady=(0, SPACING["lg"]))
        
        # Logo / Title
        title_label = ttk.Label(
            about_frame,
            text="🌙 LunaVox",
            style="Title.TLabel"
        )
        title_label.pack(anchor="w")
        
        version_label = ttk.Label(
            about_frame,
            text=f"Version {LUNAVOX_VERSION}",
            style="Muted.TLabel"
        )
        version_label.pack(anchor="w", pady=(SPACING["xs"], 0))
        
        desc_label = ttk.Label(
            about_frame,
            text="High-quality neural text-to-speech synthesis",
            style="Card.TLabel",
            wraplength=600
        )
        desc_label.pack(anchor="w", pady=(SPACING["sm"], 0))
        
        # Links
        links_frame = ttk.Frame(about_frame, style="Card.TFrame")
        links_frame.pack(anchor="w", pady=(SPACING["md"], 0))
        
        ttk.Label(
            links_frame,
            text="GitHub: github.com/Lux-Luna/LunaVox",
            style="Muted.TLabel"
        ).pack(anchor="w")
        
        # Footer
        footer = ttk.Label(
            about_frame,
            text="© 2025 LunaVox Project",
            style="Muted.TLabel"
        )
        footer.pack(anchor="w", pady=(SPACING["md"], 0))
        
    def _on_theme_change(self):
        """Handle theme change."""
        new_theme = self.theme_var.get()

        apply_theme(self.app, new_theme)
        
        # Refresh the entire UI without restart
        self.app.refresh_theme()
        
        self.app.set_status(f"Theme changed to {new_theme.capitalize()}")

        
    def _on_language_change(self):
        """Handle language change."""
        new_lang = self.lang_var.get()
        self.app.change_language(new_lang)
        
    def _update_runtime_status(self):
        """Update the runtime status display."""
        def task():
            try:
                from lunavox_tts.Utils.EnvManager import env_manager
                mode = env_manager.get_mode()
                is_gpu = env_manager.is_gpu_installed()
                
                if mode == "gpu" and is_gpu:
                    status = "🎮 GPU (CUDA)"
                    self.runtime_var.set("gpu")
                else:
                    status = "🖥️ CPU"
                    self.runtime_var.set("cpu")
                    
                self.after(0, lambda: self.runtime_status_label.configure(text=status))
            except Exception as e:
                self.after(0, lambda: self.runtime_status_label.configure(
                    text=f"Error: {str(e)[:30]}"
                ))
                
        threading.Thread(target=task, daemon=True).start()
        
    def _apply_runtime_change(self):
        """Apply runtime change (requires restart)."""
        new_mode = self.runtime_var.get()
        
        try:
            from lunavox_tts.Utils.EnvManager import env_manager
            current_mode = env_manager.get_mode()
            
            if new_mode == current_mode:
                messagebox.showinfo("Info", f"Already using {new_mode.upper()} mode.")
                return
                
            # Set the mode
            env_manager.set_mode(new_mode)
            
            # Prompt for installation if needed
            if new_mode == "gpu":
                result = messagebox.askyesno(
                    "GPU Runtime",
                    "Switch to GPU mode?\n\n"
                    "This will install CUDA dependencies (requires restart).\n"
                    "Make sure you have a compatible NVIDIA GPU."
                )
                if result:
                    env_manager.install_gpu_runtime()
                    messagebox.showinfo(
                        "Restart Required",
                        "GPU runtime installed. Please restart the application."
                    )
            else:
                env_manager.install_cpu_runtime()
                messagebox.showinfo(
                    "Restart Required",
                    "CPU runtime set. Please restart the application."
                )
                
            self._update_runtime_status()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        
    def _get_cache_size(self):
        """Calculate total cache size."""
        from main import REPO_ROOT
        
        total_size = 0
        cache_dirs = [
            REPO_ROOT / ".cache",
            Path.home() / ".cache" / "lunavox",
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
                        
        return total_size
        
    def _format_size(self, size_bytes):
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
        
    def _update_cache_size(self):
        """Update the cache size display."""
        def task():
            size = self._get_cache_size()
            self.after(0, lambda: self.cache_size_label.configure(
                text=f"Cache size: {self._format_size(size)}"
            ))
            
        threading.Thread(target=task, daemon=True).start()
        
    def _clean_cache(self):
        """Clean application cache."""
        try:
            from lunavox_tts.ModelManager import model_manager
            model_manager.clean_cache()
            
            self.app.set_status(get_text(self.app.lang, "cache_cleaned"))
            messagebox.showinfo(
                get_text(self.app.lang, "success"),
                get_text(self.app.lang, "cache_cleaned")
            )
            
            # Update cache size display
            self.cache_size_label.configure(text="Cache size: 0 B")
            self._update_cache_size()
            
        except Exception as e:
            messagebox.showerror(get_text(self.app.lang, "error"), str(e))
            
    def update_ui_texts(self):
        """Update UI text when language changes."""
        lang = self.app.lang
        self.btn_clean_cache.configure(text=get_text(lang, "clean_cache"))
