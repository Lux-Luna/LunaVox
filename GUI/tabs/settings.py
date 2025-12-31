
import tkinter as tk
from tkinter import ttk, messagebox
from i18n import I18N

class SettingsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=20)
        self.app = app
        self.build_ui()
        
    def build_ui(self):
        ttk.Label(self, text=I18N[self.app.lang]["ui_language"]).pack(anchor="w")
        self.ui_lang_cb = ttk.Combobox(self, values=["en", "zh", "ja"], state="readonly")
        self.ui_lang_cb.set(self.app.lang)
        self.ui_lang_cb.pack(fill="x", pady=5)
        self.ui_lang_cb.bind("<<ComboboxSelected>>", self.on_ui_lang_change)
        
        ttk.Button(self, text=I18N[self.app.lang]["clean_cache"], command=self.clean_cache).pack(pady=20)

    def on_ui_lang_change(self, event=None):
        new_lang = self.ui_lang_cb.get()
        self.app.change_language(new_lang)
        
    def clean_cache(self):
        from lunavox_tts.ModelManager import model_manager
        try:
            model_manager.clean_cache()
            messagebox.showinfo("Success", "Cache cleaned")
        except Exception as e:
            messagebox.showerror("Error", str(e))
