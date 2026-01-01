#!/usr/bin/env python3
"""
LunaVox GUI - Modern Speech Synthesis Interface
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Add GUI and GUI/tabs to path for easy imports
GUI_DIR = REPO_ROOT / "GUI"
TABS_DIR = GUI_DIR / "tabs"
for d in [GUI_DIR, TABS_DIR]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from i18n import I18N, get_text
from theme import apply_theme, COLORS, FONTS, SPACING
from tabs.inference import InferenceTab
from tabs.conversion import ConversionTab
from tabs.persona import PersonaTab
from tabs.settings import SettingsTab


class LunaVoxGUI(tk.Tk):
    """Main application window for LunaVox TTS GUI."""
    
    def __init__(self):
        super().__init__()
        
        # State
        self.lang = "en"
        self.loaded_character = None
        self.loaded_persona = None
        
        # Window setup
        self.title(get_text(self.lang, "title"))
        self.geometry("1000x750")
        self.minsize(800, 600)
        
        # Apply theme
        apply_theme(self)
        
        # Build UI
        self._build_ui()
        
    def _build_ui(self):
        """Construct the main UI layout."""
        
        # Header
        header_frame = ttk.Frame(self, style="TFrame")
        header_frame.pack(fill="x", padx=SPACING["lg"], pady=(SPACING["lg"], SPACING["sm"]))
        
        title_label = ttk.Label(
            header_frame, 
            text="🌙 LunaVox",
            style="Title.TLabel"
        )
        title_label.pack(side="left")
        
        # Model status indicator
        self.model_status_var = tk.StringVar(value=get_text(self.lang, "no_model"))
        self.model_status_label = ttk.Label(
            header_frame,
            textvariable=self.model_status_var,
            style="Muted.TLabel"
        )
        self.model_status_label.pack(side="right", padx=SPACING["md"])
        
        # Main notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=SPACING["lg"], pady=SPACING["sm"])
        
        # Initialize tabs
        self.inf_tab = InferenceTab(self.notebook, self)
        self.conv_tab = ConversionTab(self.notebook, self)
        self.persona_tab = PersonaTab(self.notebook, self)
        self.set_tab = SettingsTab(self.notebook, self)
        
        # Add tabs to notebook
        self.notebook.add(self.inf_tab, text=get_text(self.lang, "tab_inference"))
        self.notebook.add(self.persona_tab, text=get_text(self.lang, "tab_persona"))
        self.notebook.add(self.conv_tab, text=get_text(self.lang, "tab_conversion"))
        self.notebook.add(self.set_tab, text=get_text(self.lang, "tab_settings"))
        
        # Status bar
        self._build_status_bar()
        
    def _build_status_bar(self):
        """Build the bottom status bar."""
        status_frame = tk.Frame(self, bg=COLORS["bg_surface"], height=32)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value=get_text(self.lang, "status_ready"))
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=COLORS["bg_surface"],
            fg=COLORS["text_secondary"],
            font=(FONTS["family"], FONTS["size_sm"]),
            anchor="w",
            padx=SPACING["lg"]
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        
        # Progress bar (hidden by default)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            mode="indeterminate",
            length=120
        )
        
    def set_status(self, msg: str, show_progress: bool = False):
        """Update the status bar message."""
        self.status_var.set(msg)
        
        if show_progress:
            self.progress_bar.pack(side="right", padx=SPACING["lg"], pady=SPACING["xs"])
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            
        self.update_idletasks()
        
    def update_model_status(self, character: str = None, persona: str = None):
        """Update the model status indicator in header."""
        if character:
            self.loaded_character = character
            if persona:
                self.loaded_persona = persona
                status = f"✓ {character} ({persona})"
            else:
                status = f"✓ {character}"
            self.model_status_var.set(status)
            self.model_status_label.configure(foreground=COLORS["success"])
        else:
            self.loaded_character = None
            self.loaded_persona = None
            self.model_status_var.set(get_text(self.lang, "no_model"))
            self.model_status_label.configure(foreground=COLORS["text_muted"])
            
    def change_language(self, new_lang: str):
        """Change the UI language."""
        self.lang = new_lang
        self.title(get_text(self.lang, "title"))
        
        # Update tab headers
        self.notebook.tab(0, text=get_text(self.lang, "tab_inference"))
        self.notebook.tab(1, text=get_text(self.lang, "tab_persona"))
        self.notebook.tab(2, text=get_text(self.lang, "tab_conversion"))
        self.notebook.tab(3, text=get_text(self.lang, "tab_settings"))
        
        # Update status
        self.status_var.set(get_text(self.lang, "status_ready"))
        
        # Update model status if no model loaded
        if not self.loaded_character:
            self.model_status_var.set(get_text(self.lang, "no_model"))
        
        # Notify tabs to update their text
        for tab in [self.inf_tab, self.conv_tab, self.persona_tab, self.set_tab]:
            if hasattr(tab, 'update_ui_texts'):
                tab.update_ui_texts()
    
    def refresh_theme(self):
        """Refresh the theme across all widgets without restart."""
        from theme import apply_theme, COLORS, get_text_widget_config, get_listbox_config
        
        # Re-apply the theme to update all ttk styles
        apply_theme(self)
        
        # Update native tk widgets that don't use ttk styles
        self._refresh_tk_widgets(self)
        
        # Notify tabs to refresh their widgets
        for tab in [self.inf_tab, self.conv_tab, self.persona_tab, self.set_tab]:
            if hasattr(tab, 'refresh_theme'):
                tab.refresh_theme()
            self._refresh_tk_widgets(tab)
                
        self.update_idletasks()
        
    def _refresh_tk_widgets(self, parent):
        """Recursively refresh native tk widgets with current theme colors."""
        from theme import COLORS, FONTS, get_text_widget_config, get_listbox_config
        
        for widget in parent.winfo_children():
            widget_class = widget.winfo_class()
            
            # Update tk.Frame
            if widget_class == 'Frame':
                try:
                    widget.configure(bg=COLORS["bg_surface"])
                except tk.TclError:
                    pass
                    
            # Update tk.Label
            elif widget_class == 'Label':
                try:
                    widget.configure(
                        bg=COLORS["bg_surface"],
                        fg=COLORS["text_secondary"]
                    )
                except tk.TclError:
                    pass
                    
            # Update tk.Text
            elif widget_class == 'Text':
                config = get_text_widget_config()
                try:
                    widget.configure(**config)
                except tk.TclError:
                    pass
                    
            # Update tk.Listbox
            elif widget_class == 'Listbox':
                config = get_listbox_config()
                try:
                    widget.configure(**config)
                except tk.TclError:
                    pass
                    
            # Update tk.Canvas
            elif widget_class == 'Canvas':
                try:
                    widget.configure(bg=COLORS["bg_dark"])
                except tk.TclError:
                    pass
            
            # Recurse into children
            self._refresh_tk_widgets(widget)



def main():
    """Entry point for the GUI application."""
    app = LunaVoxGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
