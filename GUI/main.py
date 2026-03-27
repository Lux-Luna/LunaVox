import customtkinter as ctk
import os
from engine import LunaVoxEngine
from i18n import TRANSLATIONS
import threading

class LunaVoxGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.lang = "en"
        self.engine = LunaVoxEngine(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models = self.engine.discover_models()
        self.current_model = None

        self.setup_ui()
        self.update_texts()

    def setup_ui(self):
        self.title("LunaVox TTS")
        self.geometry("800x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Main Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header - Backend Info & Language Switch
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.backend_label = ctk.CTkLabel(self.header_frame, text="", font=ctk.CTkFont(size=12))
        self.backend_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.lang_btn = ctk.CTkButton(self.header_frame, text="中文", width=80, command=self.toggle_language)
        self.lang_btn.grid(row=0, column=1, padx=10, pady=10)

        # Content - Scrollable Frame
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Model Selection
        self.model_frame = ctk.CTkFrame(self.scroll_frame)
        self.model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.model_label = ctk.CTkLabel(self.model_frame, text="Model", font=ctk.CTkFont(weight="bold"))
        self.model_label.pack(side="top", padx=10, pady=5, anchor="w")
        
        model_names = [m["name"] for m in self.models]
        self.model_dropdown = ctk.CTkOptionMenu(self.model_frame, values=model_names, command=self.on_model_change)
        self.model_dropdown.pack(side="top", padx=10, pady=5, fill="x")

        # Parameters Form
        self.form_frame = ctk.CTkFrame(self.scroll_frame)
        self.form_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.form_frame.grid_columnconfigure(0, weight=1)

        # Sentence
        self.text_label = ctk.CTkLabel(self.form_frame, text="Sentence")
        self.text_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.text_input = ctk.CTkEntry(self.form_frame, placeholder_text="Enter text to synthesize...")
        self.text_input.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.text_input.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # Dynamic Fields
        self.instruct_label = ctk.CTkLabel(self.form_frame, text="Instruct")
        self.instruct_input = ctk.CTkTextbox(self.form_frame, height=80)
        self.instruct_input.bind("<KeyRelease>", lambda _: self.update_command_preview())
        
        self.speaker_label = ctk.CTkLabel(self.form_frame, text="Speaker")
        self.speaker_dropdown = ctk.CTkOptionMenu(self.form_frame, values=[])

        # Reference selection (dropdown + browse)
        self.ref_label = ctk.CTkLabel(self.form_frame, text="Reference")
        self.ref_action_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.ref_dropdown = ctk.CTkOptionMenu(self.ref_action_frame, values=self.engine.discover_references(), command=lambda _: self.update_command_preview())
        self.ref_browse_btn = ctk.CTkButton(self.ref_action_frame, text="Browse...", width=100, command=self.browse_reference)
        
        self.ref_dropdown.pack(side="left", padx=(0, 10), fill="x", expand=True)
        self.ref_browse_btn.pack(side="left")

        self.ref_text_label = ctk.CTkLabel(self.form_frame, text="Reference Text")
        self.ref_text_input = ctk.CTkEntry(self.form_frame, placeholder_text="Label for the reference audio...")
        self.ref_text_input.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # Language
        self.lang_sel_label = ctk.CTkLabel(self.form_frame, text="Language")
        self.lang_sel_dropdown = ctk.CTkOptionMenu(self.form_frame, values=["auto"], command=lambda _: self.update_command_preview())

        # Advanced Section (Collapsible)
        self.adv_btn = ctk.CTkButton(self.scroll_frame, text="+ Advanced Parameters", fg_color="transparent", border_width=1, command=self.toggle_advanced)
        self.adv_btn.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        
        self.adv_frame = ctk.CTkFrame(self.scroll_frame)
        self.adv_visible = False
        
        # Sliders for Temp
        self.temp_label = ctk.CTkLabel(self.adv_frame, text="Temperature: 0.6")
        self.temp_slider = ctk.CTkSlider(self.adv_frame, from_=0.1, to=1.5, number_of_steps=14, 
                                        command=self._on_temp_change)
        self.temp_slider.set(0.6)

        self.pred_temp_label = ctk.CTkLabel(self.adv_frame, text="Predictor Temp: 0.6")
        self.pred_temp_slider = ctk.CTkSlider(self.adv_frame, from_=0.1, to=1.5, number_of_steps=14, 
                                             command=self._on_pred_temp_change)
        self.pred_temp_slider.set(0.6)

        self.seed_label = ctk.CTkLabel(self.adv_frame, text="Seed")
        self.seed_input = ctk.CTkEntry(self.adv_frame)
        self.seed_input.insert(0, "42")

        self.tokens_label = ctk.CTkLabel(self.adv_frame, text="Max Tokens")
        self.tokens_input = ctk.CTkEntry(self.adv_frame)
        self.tokens_input.insert(0, "400")
        for entry in [self.seed_input, self.tokens_input]:
            entry.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # top-k, top-p, penalty, threads
        self.extra_adv_frame = ctk.CTkFrame(self.adv_frame, fg_color="transparent")
        
        self.top_k_label = ctk.CTkLabel(self.extra_adv_frame, text="Top-K")
        self.top_k_input = ctk.CTkEntry(self.extra_adv_frame, width=100)
        self.top_k_input.insert(0, "50")

        self.top_p_label = ctk.CTkLabel(self.extra_adv_frame, text="Top-P")
        self.top_p_input = ctk.CTkEntry(self.extra_adv_frame, width=100)
        self.top_p_input.insert(0, "1.0")

        self.penalty_label = ctk.CTkLabel(self.extra_adv_frame, text="Penalty")
        self.penalty_input = ctk.CTkEntry(self.extra_adv_frame, width=100)
        self.penalty_input.insert(0, "1.05")

        self.threads_label = ctk.CTkLabel(self.extra_adv_frame, text="Threads")
        self.threads_input = ctk.CTkEntry(self.extra_adv_frame, width=100)
        self.threads_input.insert(0, "4")

        # Command Preview Section
        self.cmd_preview_btn = ctk.CTkButton(self.scroll_frame, text="+ Command Preview", fg_color="transparent", border_width=1, command=self.toggle_command_preview)
        self.cmd_preview_btn.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        
        self.cmd_preview_frame = ctk.CTkFrame(self.scroll_frame)
        self.cmd_preview_visible = False
        self.cmd_preview_box = ctk.CTkTextbox(self.cmd_preview_frame, height=100, font=ctk.CTkFont(family="Consolas", size=12))
        self.cmd_preview_box.configure(state="disabled")

        for entry in [self.top_k_input, self.top_p_input, self.penalty_input, self.threads_input]:
            entry.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # Footer
        self.footer_frame = ctk.CTkFrame(self)
        self.footer_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        self.generate_btn = ctk.CTkButton(self.footer_frame, text="Generate", font=ctk.CTkFont(size=16, weight="bold"), height=50, command=self.generate)
        self.generate_btn.pack(side="top", padx=20, pady=10, fill="x")

        self.status_label = ctk.CTkLabel(self.footer_frame, text="Ready")
        self.status_label.pack(side="top", padx=20, pady=5)

        # Initialize Dropdown
        if model_names:
            self.model_dropdown.set(model_names[0])
            self.on_model_change(model_names[0])

        self.show_backend_info()

    def _on_temp_change(self, v):
        self.temp_label.configure(text=f"{self.t('temperature')}: {round(v, 2)}")
        self.update_command_preview()

    def _on_pred_temp_change(self, v):
        self.pred_temp_label.configure(text=f"{self.t('predictor_temp')}: {round(v, 2)}")
        self.update_command_preview()

    def t(self, key):
        return TRANSLATIONS[self.lang].get(key, key)

    def toggle_language(self):
        self.lang = "zh" if self.lang == "en" else "en"
        self.update_texts()

    def update_texts(self):
        self.title(self.t("title"))
        self.lang_btn.configure(text=self.t("language_switch"))
        self.model_label.configure(text=self.t("model_selection"))
        self.text_label.configure(text=self.t("sentence"))
        self.instruct_label.configure(text=self.t("instruct"))
        self.speaker_label.configure(text=self.t("speaker"))
        self.ref_label.configure(text=self.t("reference"))
        self.ref_text_label.configure(text=self.t("ref_text"))
        self.lang_sel_label.configure(text=self.t("language"))
        self.adv_btn.configure(text=f"{'+' if not self.adv_visible else '-'} {self.t('advanced_parameters')}")
        self.temp_label.configure(text=f"{self.t('temperature')}: {round(self.temp_slider.get(), 2)}")
        self.pred_temp_label.configure(text=f"{self.t('predictor_temp')}: {round(self.pred_temp_slider.get(), 2)}")
        self.seed_label.configure(text=self.t("seed"))
        self.tokens_label.configure(text=self.t("max_new_tokens"))
        self.top_k_label.configure(text=self.t("top_k"))
        self.top_p_label.configure(text=self.t("top_p"))
        self.penalty_label.configure(text=self.t("repetition_penalty"))
        self.threads_label.configure(text=self.t("threads"))
        self.cmd_preview_btn.configure(text=f"{'+' if not self.cmd_preview_visible else '-'} {self.t('command_preview')}")
        self.generate_btn.configure(text=self.t("generate"))
        self.browse_reference_btn = self.ref_browse_btn.configure(text=self.t("browse"))
        self.status_label.configure(text=self.t("status_idle" if self.status_label.cget("text") in ["Ready", "待命"] else "status_idle")) # Simplistic
        self.show_backend_info()

    def show_backend_info(self):
        info = self.engine.get_backend_info()
        if info:
            onnx = info.get("onnx", {}).get("provider", "CPU")
            llama = info.get("llama", {}).get("backend", "cpu")
            self.backend_label.configure(text=f"ONNX: {onnx} | Llama: {llama}")
        else:
            self.backend_label.configure(text="No metadata found")

    def on_model_change(self, name):
        self.current_model = next(m for m in self.models if m["name"] == name)
        mtype = self.current_model["type"]
        
        # Clear dynamic fields
        for widget in [self.instruct_label, self.instruct_input, self.speaker_label, self.speaker_dropdown, 
                       self.ref_label, self.ref_dropdown, self.ref_text_label, self.ref_text_input,
                       self.lang_sel_label, self.lang_sel_dropdown]:
            widget.grid_forget()

        row = 2
        # Language selection is always available
        self.lang_sel_label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
        self.lang_sel_dropdown.grid(row=row+1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        # Populate languages
        langs = ["auto"] + [l for l in self.current_model["profile"].get("language_names", []) if len(l) > 2]
        self.lang_sel_dropdown.configure(values=langs)
        self.lang_sel_dropdown.set("auto")
        
        row += 2
        if mtype == "base":
            self.ref_label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
            self.ref_action_frame.grid(row=row+1, column=0, padx=10, pady=(0, 10), sticky="ew")
            self.ref_text_label.grid(row=row+2, column=0, padx=10, pady=(10, 0), sticky="w")
            self.ref_text_input.grid(row=row+3, column=0, padx=10, pady=(0, 10), sticky="ew")
        elif mtype == "custom":
            self.speaker_label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
            self.speaker_dropdown.grid(row=row+1, column=0, padx=10, pady=(0, 10), sticky="ew")
            # Populate speaker dropdown
            speakers = self.current_model["profile"].get("speaker_names", [])
            self.speaker_dropdown.configure(values=speakers)
            if speakers: self.speaker_dropdown.set(speakers[0])
            
            # Check instruct support
            if self.current_model.get("instruct_support"):
                self.instruct_label.grid(row=row+2, column=0, padx=10, pady=(10, 0), sticky="w")
                self.instruct_input.grid(row=row+3, column=0, padx=10, pady=(0, 10), sticky="ew")
        elif mtype == "design":
            if self.current_model.get("instruct_support"):
                self.instruct_label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
                self.instruct_input.grid(row=row+1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.update_command_preview()

    def browse_reference(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Reference Files", "*.wav *.json")])
        if file_path:
            # Check if it's inside or outside the project for relative path
            try:
                rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.ref_dropdown.configure(values=[rel_path] + list(self.ref_dropdown.cget("values")))
                self.ref_dropdown.set(rel_path)
            except ValueError:
                self.ref_dropdown.configure(values=[file_path] + list(self.ref_dropdown.cget("values")))
                self.ref_dropdown.set(file_path)
            self.update_command_preview()

    def toggle_advanced(self):
        if self.adv_visible:
            self.adv_frame.grid_forget()
            self.adv_visible = False
        else:
            self.adv_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
            self.adv_frame.grid_columnconfigure(0, weight=1)
            
            self.temp_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            self.temp_slider.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
            self.pred_temp_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
            self.pred_temp_slider.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
            
            self.seed_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
            self.seed_input.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
            self.tokens_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
            self.tokens_input.grid(row=7, column=0, padx=10, pady=5, sticky="ew")

            self.extra_adv_frame.grid(row=8, column=0, padx=10, pady=10, sticky="ew")
            self.extra_adv_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
            self.top_k_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.top_k_input.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
            self.top_p_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")
            self.top_p_input.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self.penalty_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")
            self.penalty_input.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
            self.threads_label.grid(row=0, column=3, padx=5, pady=2, sticky="w")
            self.threads_input.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
            
            self.adv_visible = True
        self.update_texts()

    def toggle_command_preview(self):
        if self.cmd_preview_visible:
            self.cmd_preview_frame.grid_forget()
            self.cmd_preview_visible = False
        else:
            self.cmd_preview_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
            self.cmd_preview_frame.grid_columnconfigure(0, weight=1)
            self.cmd_preview_box.pack(padx=10, pady=10, fill="both", expand=True)
            self.cmd_preview_visible = True
            self.update_command_preview()
        self.update_texts()

    def update_command_preview(self):
        args = self.get_current_args()
        if not args: return
        cmd_str = self.engine.get_command_string(args)
        self.cmd_preview_box.configure(state="normal")
        self.cmd_preview_box.delete("1.0", "end")
        self.cmd_preview_box.insert("1.0", cmd_str)
        self.cmd_preview_box.configure(state="disabled")

    def get_current_args(self):
        if not self.current_model: return None
        try:
            args = {
                "model_id": self.current_model["id"],
                "model_path": self.current_model["path"],
                "model_type": self.current_model["type"],
                "text": self.text_input.get() or "...",
                "temperature": self.temp_slider.get(),
                "predictor_temp": self.pred_temp_slider.get(),
                "max_new_tokens": int(self.tokens_input.get() or 400),
                "seed": int(self.seed_input.get() or 42),
                "language": self.lang_sel_dropdown.get(),
                "top_k": int(self.top_k_input.get() or 50),
                "top_p": float(self.top_p_input.get() or 1.0),
                "repetition_penalty": float(self.penalty_input.get() or 1.05),
                "threads": int(self.threads_input.get() or 4)
            }
            if args["model_type"] == "base":
                args["reference"] = self.ref_dropdown.get()
                args["ref_text"] = self.ref_text_input.get()
            elif args["model_type"] == "custom":
                args["speaker"] = self.speaker_dropdown.get()
                args["instruct"] = self.instruct_input.get("1.0", "end-1c")
            elif args["model_type"] == "design":
                args["instruct"] = self.instruct_input.get("1.0", "end-1c")
            return args
        except ValueError:
            return None

    def generate(self):
        args = self.get_current_args()
        if not args or args["text"] == "...":
            self.status_label.configure(text=self.t("sentence"), text_color="red")
            return
        elif args["model_type"] == "custom":
            args["speaker"] = self.speaker_dropdown.get()
            args["instruct"] = self.instruct_input.get("1.0", "end-1c")
        elif args["model_type"] == "design":
            args["instruct"] = self.instruct_input.get("1.0", "end-1c")
            if not args["instruct"]:
                self.status_label.configure(text=self.t("instruct"), text_color="red")
                return

        self.status_label.configure(text=self.t("status_running"), text_color="white")
        self.generate_btn.configure(state="disabled")

        def run():
            proc = self.engine.run_synthesis(args)
            stdout, stderr = proc.communicate()
            if proc.returncode == 0:
                self.after(0, lambda: self.status_label.configure(text=self.t("status_success"), text_color="green"))
            else:
                self.after(0, lambda: self.status_label.configure(text=f"{self.t('status_error')}: {stderr}", text_color="red"))
            self.after(0, lambda: self.generate_btn.configure(state="normal"))

        threading.Thread(target=run).start()

if __name__ == "__main__":
    app = LunaVoxGUI()
    app.mainloop()
