
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

from i18n import I18N
from tabs.inference import InferenceTab
from tabs.conversion import ConversionTab
from tabs.settings import SettingsTab

class LunaVoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.lang = "en"
        self.loaded_character = None
        
        # Setup window
        self.title(I18N[self.lang]["title"])
        self.geometry("950x800")
        self.configure(bg="#f0f2f5")
        
        # Style
        self.setup_style()
        
        # Main container
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize Tabs
        self.inf_tab = InferenceTab(self.notebook, self)
        self.conv_tab = ConversionTab(self.notebook, self)
        self.set_tab = SettingsTab(self.notebook, self)
        
        self.notebook.add(self.inf_tab, text=I18N[self.lang]["tab_inference"])
        self.notebook.add(self.conv_tab, text=I18N[self.lang]["tab_conversion"])
        self.notebook.add(self.set_tab, text=I18N[self.lang]["tab_settings"])
        
        # Status Bar
        self.status_var = tk.StringVar(value=I18N[self.lang]["status_ready"])
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_color = "#f0f2f5"
        primary_color = "#4a90e2"
        text_color = "#333333"
        
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=text_color, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Primary.TButton", background=primary_color, foreground="white")
        style.map("Primary.TButton", background=[('active', '#357abd')])
        
        style.configure("TNotebook", background=bg_color)
        style.configure("TNotebook.Tab", padding=[12, 6], font=("Segoe UI", 10))

    def set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def change_language(self, new_lang):
        self.lang = new_lang
        self.title(I18N[self.lang]["title"])
        
        # Update Tab headers
        self.notebook.tab(0, text=I18N[self.lang]["tab_inference"])
        self.notebook.tab(1, text=I18N[self.lang]["tab_conversion"])
        self.notebook.tab(2, text=I18N[self.lang]["tab_settings"])
        
        self.status_var.set(I18N[self.lang]["status_ready"])
        
        # To fully refresh UI components without restart, we would need to 
        # call an update_ui_texts() on each tab.
        if hasattr(self.inf_tab, 'update_ui_texts'):
            self.inf_tab.update_ui_texts()
            
        messagebox.showinfo("LunaVox", "Language updated to " + new_lang.upper())

if __name__ == "__main__":
    app = LunaVoxGUI()
    app.mainloop()
