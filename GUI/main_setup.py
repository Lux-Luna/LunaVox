import customtkinter as ctk
import os
import sys

# Add current dir to path to import components
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path: sys.path.append(cur_dir)

from engine import LunaVoxEngine
from i18n import TRANSLATIONS
from components.setup_page import SetupPage

class LunaVoxSetup(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.lang = "en"
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine = LunaVoxEngine(self.root_dir)

        self.setup_ui()
        self.update_texts()

    def setup_ui(self):
        self.title("LunaVox Setup & Tools")
        self.geometry("800x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # In Setup GUI, the SetupPage is the main content.
        # It already contains its own language dropdown for its internal text.
        self.setup_page = SetupPage(
            self, self.t,
            on_back=self.on_exit,
            on_lang_change=self.set_language,
            on_refresh=self.on_refresh
        )
        self.setup_page.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    def t(self, key):
        return TRANSLATIONS[self.lang].get(key, key)

    def set_language(self, new_lang):
        self.lang = new_lang
        self.update_texts()
        self.setup_page.update_texts()
        # Sync the setup page's internal dropdown
        self.setup_page.set_lang_dropdown(self.lang)

    def update_texts(self):
        self.title(f"{self.t('setup_btn')} - LunaVox")

    def on_refresh(self):
        print("Setup: Refresh triggered")

    def on_exit(self):
        self.destroy()

if __name__ == "__main__":
    app = LunaVoxSetup()
    app.mainloop()
