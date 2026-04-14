import customtkinter as ctk
import os
import sys
import threading
from pathlib import Path

# Add current dir to path to import components
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path: sys.path.append(cur_dir)

from engine import LunaVoxEngine
from i18n import TRANSLATIONS
from components.header import HeaderFrame
from components.report import ReportFrame
from lunavox.core.project import resolve_project_root

class LunaVoxTTS(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.lang = "en"
        # Resolve the LunaVox project root via the canonical helper so
        # GUI / CLI / tests all agree on where models and build/ live.
        self.root_dir = str(resolve_project_root(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.engine = LunaVoxEngine(self.root_dir)
        self.models = self.engine.discover_models()
        self.current_model = None
        self.platform = "Windows"

        self.setup_ui()
        self.update_texts()
        # Release the C++ engine cleanly on window close so the DLL is
        # not left holding mmap'd model files when the process exits.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass
        self.destroy()

    def setup_ui(self):
        self.title("LunaVox TTS")
        self.geometry("950x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header: show_setup=False, setup gear is replaced by language toggle
        self.header = HeaderFrame(
            self, self.t, 
            on_lang_change=self.on_lang_toggle, 
            on_platform_change=self.on_platform_change,
            show_setup=False
        )
        self.header.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="ew")

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # 1. Model Selection (Row 0)
        self.config_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.config_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.config_frame.grid_columnconfigure(0, weight=1)

        self.model_frame = ctk.CTkFrame(self.config_frame)
        self.model_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.model_label = ctk.CTkLabel(self.model_frame, text="Model Selection", font=ctk.CTkFont(size=12, weight="bold"))
        self.model_label.pack(side="top", anchor="w", padx=10, pady=(8, 0))
        model_names = [m["name"] for m in self.models]
        self.model_dropdown = ctk.CTkOptionMenu(self.model_frame, values=model_names, command=self.on_model_change, height=35)
        self.model_dropdown.pack(side="top", fill="x", padx=10, pady=(2, 4))
        self.model_desc_label = ctk.CTkLabel(self.model_frame, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#5BA0D0")
        self.model_desc_label.pack(side="top", anchor="w", padx=12, pady=(0, 8))

        # 2. Audio/Voice Panel (Row 1)
        self.audio_cfg_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.audio_cfg_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.lang_sel_label = ctk.CTkLabel(self.audio_cfg_frame, text="Language", font=ctk.CTkFont(size=12, weight="bold"))
        self.lang_sel_dropdown = ctk.CTkOptionMenu(self.audio_cfg_frame, values=["auto"], command=lambda _: self.update_command_preview(), width=160)

        self.speaker_label = ctk.CTkLabel(self.audio_cfg_frame, text="Speaker", font=ctk.CTkFont(size=12, weight="bold"))
        self.speaker_dropdown = ctk.CTkOptionMenu(self.audio_cfg_frame, values=[], width=280)
        
        self.ref_label = ctk.CTkLabel(self.audio_cfg_frame, text="Reference Audio", font=ctk.CTkFont(size=12, weight="bold"))
        self.ref_action_frame = ctk.CTkFrame(self.audio_cfg_frame, fg_color="transparent")
        self.ref_dropdown = ctk.CTkOptionMenu(self.ref_action_frame, values=self.engine.discover_references(), command=lambda _: self.update_command_preview(), width=180)
        self.ref_browse_btn = ctk.CTkButton(self.ref_action_frame, text="Browse", width=80, height=26, command=self.browse_reference)
        self.ref_dropdown.pack(side="left", padx=(0, 5))
        self.ref_browse_btn.pack(side="left")

        self.ref_text_label = ctk.CTkLabel(self.audio_cfg_frame, text="Reference Transcription", font=ctk.CTkFont(size=12, weight="bold"))
        self.ref_text_input = ctk.CTkTextbox(self.audio_cfg_frame, height=65)
        self.ref_text_input.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # 3. Content Panel (Row 2)
        self.content_frame = ctk.CTkFrame(self.scroll_frame)
        self.content_frame.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.text_label = ctk.CTkLabel(self.content_frame, text="Synthesis Text", font=ctk.CTkFont(size=13, weight="bold"))
        self.text_label.grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")
        self.text_input = ctk.CTkEntry(self.content_frame, placeholder_text="Type text to synthesize...", height=45)
        self.text_input.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
        self.text_input.bind("<KeyRelease>", lambda _: self.update_command_preview())

        self.instruct_label = ctk.CTkLabel(self.content_frame, text="Instruct / Description", font=ctk.CTkFont(size=12, weight="bold"))
        self.instruct_input = ctk.CTkTextbox(self.content_frame, height=80)
        self.instruct_input.bind("<KeyRelease>", lambda _: self.update_command_preview())

        # 4. Buttons Panel (Row 3)
        self.toggle_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.toggle_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")
        self.adv_btn = ctk.CTkButton(self.toggle_frame, text="+ Advanced Settings", fg_color="transparent", border_width=1, command=self.toggle_advanced)
        self.adv_btn.pack(side="left", padx=5, fill="x", expand=True)

        # 5. Expandable Sections
        self.adv_frame = ctk.CTkFrame(self.scroll_frame)
        self.adv_visible = False
        self._setup_advanced_fields()

        # 5. Report Section
        self.report = ReportFrame(self.scroll_frame, self.t)
        self.report.grid(row=5, column=0, padx=15, pady=15, sticky="ew")

        # 6. Command Preview
        self.cmd_preview_frame = ctk.CTkFrame(self.scroll_frame)
        self.cmd_preview_frame.grid(row=6, column=0, padx=15, pady=10, sticky="ew")
        self.cmd_preview_frame.grid_columnconfigure(0, weight=1)
        self.cmd_preview_box = ctk.CTkTextbox(self.cmd_preview_frame, height=130, font=ctk.CTkFont(family="Consolas", size=11))
        self.cmd_preview_box.configure(state="disabled")
        self.cmd_preview_box.pack(padx=10, pady=10, fill="both", expand=True)

        # Footer
        self.footer_frame = ctk.CTkFrame(self)
        self.footer_frame.grid(row=2, column=0, padx=20, pady=(5, 15), sticky="ew")
        self.generate_btn = ctk.CTkButton(
            self.footer_frame, text="Generate Speech",
            font=ctk.CTkFont(size=16, weight="bold"), height=50,
            corner_radius=10, fg_color="#1F6AA5", hover_color="#144870",
            command=self.generate
        )
        self.generate_btn.pack(side="top", padx=20, pady=(10, 5), fill="x")
        self.status_label = ctk.CTkLabel(self.footer_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="top", padx=20, pady=(0, 5))

        # Initialize
        if model_names:
            self.model_dropdown.set(model_names[0])
            self.on_model_change(model_names[0])
        self.header.update_info(self.engine.get_backend_info())

    def _setup_advanced_fields(self):
        self.adv_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.temp_label = ctk.CTkLabel(self.adv_frame, text="Temperature: 0.6")
        self.temp_slider = ctk.CTkSlider(self.adv_frame, from_=0.1, to=1.5, number_of_steps=14, command=self._on_temp_change)
        self.temp_slider.set(0.6)
        self.pred_temp_label = ctk.CTkLabel(self.adv_frame, text="Predictor Temp: 0.6")
        self.pred_temp_slider = ctk.CTkSlider(self.adv_frame, from_=0.1, to=1.5, number_of_steps=14, command=self._on_pred_temp_change)
        self.pred_temp_slider.set(0.6)
        self.temp_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.temp_slider.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.pred_temp_label.grid(row=0, column=2, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.pred_temp_slider.grid(row=1, column=2, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        self.seed_label = ctk.CTkLabel(self.adv_frame, text="Seed")
        self.seed_input = ctk.CTkEntry(self.adv_frame)
        self.seed_input.insert(0, "42")
        self.tokens_label = ctk.CTkLabel(self.adv_frame, text="Max Tokens")
        self.tokens_input = ctk.CTkEntry(self.adv_frame)
        self.tokens_input.insert(0, "400")
        self.seed_label.grid(row=2, column=0, padx=10, sticky="w")
        self.seed_input.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.tokens_label.grid(row=2, column=1, padx=10, sticky="w")
        self.tokens_input.grid(row=3, column=1, padx=10, pady=(0, 10), sticky="ew")

        self.top_k_label = ctk.CTkLabel(self.adv_frame, text="Top-K")
        self.top_k_input = ctk.CTkEntry(self.adv_frame, width=80) 
        self.top_k_input.insert(0, "50")
        self.top_p_label = ctk.CTkLabel(self.adv_frame, text="Top-P")
        self.top_p_input = ctk.CTkEntry(self.adv_frame, width=80)
        self.top_p_input.insert(0, "1.0")
        self.penalty_label = ctk.CTkLabel(self.adv_frame, text="Penalty")
        self.penalty_input = ctk.CTkEntry(self.adv_frame, width=80)
        self.penalty_input.insert(0, "1.05")
        self.threads_label = ctk.CTkLabel(self.adv_frame, text="Threads")
        self.threads_input = ctk.CTkEntry(self.adv_frame, width=80)
        self.threads_input.insert(0, "4")
        self.top_k_label.grid(row=4, column=0, padx=10, pady=(5, 0), sticky="w")
        self.top_k_input.grid(row=5, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.top_p_label.grid(row=4, column=1, padx=10, pady=(5, 0), sticky="w")
        self.top_p_input.grid(row=5, column=1, padx=10, pady=(0, 10), sticky="ew")
        self.penalty_label.grid(row=4, column=2, padx=10, pady=(5, 0), sticky="w")
        self.penalty_input.grid(row=5, column=2, padx=10, pady=(0, 10), sticky="ew")
        self.threads_label.grid(row=4, column=3, padx=10, pady=(5, 0), sticky="w")
        self.threads_input.grid(row=5, column=3, padx=10, pady=(0, 10), sticky="ew")

        for ent in [self.seed_input, self.tokens_input, self.top_k_input, self.top_p_input, self.penalty_input, self.threads_input]:
            ent.bind("<KeyRelease>", lambda _: self.update_command_preview())

    def t(self, key):
        return TRANSLATIONS[self.lang].get(key, key)

    def on_lang_toggle(self, value):
        new_lang = "zh" if value == "中文" else "en"
        self.lang = new_lang
        self.update_texts()
        self.header.update_texts()
        self.report.update_texts()

    def update_texts(self):
        self.title(self.t("title"))
        self.model_label.configure(text=self.t("model_selection"))
        self.text_label.configure(text=self.t("sentence"))
        self.lang_sel_label.configure(text=self.t("language"))
        self.ref_label.configure(text=self.t("reference"))
        self.ref_text_label.configure(text=self.t("ref_text"))
        self.speaker_label.configure(text=self.t("speaker"))
        
        if self.current_model:
            mtype = self.current_model.get("type", "base")
            if mtype == "design":
                self.instruct_label.configure(text=self.t("instruct_required"))
            else:
                self.instruct_label.configure(text=self.t("instruct_optional"))
        else:
            self.instruct_label.configure(text=self.t("instruct"))
            
        self._update_model_desc()
        self.adv_btn.configure(text=f"{'+' if not self.adv_visible else '-'} {self.t('advanced_parameters')}")
        self.generate_btn.configure(text=self.t("generate"))
        self.status_label.configure(text=self.t("status_idle"))
        self.ref_browse_btn.configure(text=self.t("browse"))
        self._on_temp_change(self.temp_slider.get())
        self._on_pred_temp_change(self.pred_temp_slider.get())
        self.header.set_lang_dropdown(self.lang)

    def _on_temp_change(self, v):
        self.temp_label.configure(text=f"{self.t('temperature')}: {round(v, 2)}")
        self.update_command_preview()

    def _on_pred_temp_change(self, v):
        self.pred_temp_label.configure(text=f"{self.t('predictor_temp')}: {round(v, 2)}")
        self.update_command_preview()

    def browse_reference(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Reference", "*.wav *.json")])
        if file_path:
            try:
                rel_path = os.path.relpath(file_path, self.engine.root_dir)
                self.ref_dropdown.configure(values=[rel_path] + list(self.ref_dropdown.cget("values")))
                self.ref_dropdown.set(rel_path)
            except ValueError:
                self.ref_dropdown.set(file_path)
            self.update_command_preview()

    def _update_model_desc(self):
        if not self.current_model:
            self.model_desc_label.configure(text="")
            return
        mtype = self.current_model.get("type", "base")
        desc_key = f"model_desc_{mtype}"
        self.model_desc_label.configure(text=self.t(desc_key))

    def on_model_change(self, name):
        self.current_model = next(m for m in self.models if m["name"] == name)
        profile = self.current_model.get("profile", {})
        mtype = self.current_model.get("type", "base")
        self._update_model_desc()
        
        widgets = [self.lang_sel_label, self.lang_sel_dropdown, self.speaker_label, self.speaker_dropdown,
                   self.ref_label, self.ref_action_frame, self.ref_text_label, self.ref_text_input,
                   self.instruct_label, self.instruct_input]
        for w in widgets: w.grid_forget()

        cp = 10
        self.lang_sel_label.configure(text=self.t("language"))
        self.lang_sel_label.grid(row=0, column=0, padx=cp, pady=(5, 0), sticky="w")
        self.lang_sel_dropdown.grid(row=1, column=0, padx=cp, pady=(0, 5), sticky="ew")

        if mtype == "base":
            self.audio_cfg_frame.grid_columnconfigure((0, 1, 2), weight=1)
            self.ref_label.grid(row=0, column=1, padx=cp, pady=(5, 0), sticky="w")
            self.ref_action_frame.grid(row=1, column=1, padx=cp, pady=(0, 5), sticky="ew")
            self.ref_text_label.grid(row=0, column=2, padx=cp, pady=(5, 0), sticky="w")
            self.ref_text_input.grid(row=1, column=2, padx=cp, pady=(0, 5), sticky="nsew")
            self.ref_text_input.configure(height=60) 
        elif mtype == "custom":
            self.audio_cfg_frame.grid_columnconfigure((0, 1), weight=1)
            self.audio_cfg_frame.grid_columnconfigure(2, weight=0)
            speakers = profile.get("speaker_names", [])
            if speakers:
                self.speaker_label.grid(row=0, column=1, padx=cp, pady=(5, 0), sticky="w")
                self.speaker_dropdown.grid(row=1, column=1, padx=cp, pady=(0, 5), sticky="ew")
                self.speaker_dropdown.configure(values=speakers)
                self.speaker_dropdown.set(speakers[0])
        else:
            self.audio_cfg_frame.grid_columnconfigure(0, weight=1)
            self.audio_cfg_frame.grid_columnconfigure((1, 2), weight=0)

        if self.current_model.get("instruct_support"):
            if mtype == "design":
                self.instruct_label.configure(text=self.t("instruct_required"))
            else:
                self.instruct_label.configure(text=self.t("instruct_optional"))
            self.instruct_label.grid(row=2, column=0, padx=15, pady=(5, 2), sticky="w")
            self.instruct_input.grid(row=3, column=0, padx=15, pady=(0, 10), sticky="ew")

        langs = ["auto"] + [l for l in profile.get("language_names", []) if len(l) > 2]
        self.lang_sel_dropdown.configure(values=langs)
        self.lang_sel_dropdown.set("auto")
        self.update_command_preview()

    def toggle_advanced(self):
        if self.adv_visible:
            self.adv_frame.grid_forget()
            self.adv_visible = False
        else:
            self.adv_frame.grid(row=4, column=0, padx=15, pady=10, sticky="ew")
            self.adv_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
            self.temp_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            self.temp_slider.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            self.pred_temp_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            self.pred_temp_slider.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            self.seed_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
            self.seed_input.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
            self.tokens_label.grid(row=4, column=1, padx=10, pady=5, sticky="w")
            self.tokens_input.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
            self.top_k_label.grid(row=6, column=0, padx=10)
            self.top_k_input.grid(row=7, column=0, padx=10, pady=5)
            self.top_p_label.grid(row=6, column=1, padx=10)
            self.top_p_input.grid(row=7, column=1, padx=10, pady=5)
            self.penalty_label.grid(row=6, column=2, padx=10)
            self.penalty_input.grid(row=7, column=2, padx=10, pady=5)
            self.threads_label.grid(row=6, column=3, padx=10)
            self.threads_input.grid(row=7, column=3, padx=10, pady=5)
            self.adv_visible = True
        self.update_texts()

    def update_command_preview(self):
        args = self.get_current_args()
        if not args: return
        args["platform"] = self.platform
        cmd_str = self.engine.get_command_string(args)
        self.cmd_preview_box.configure(state="normal")
        self.cmd_preview_box.delete("1.0", "end")
        self.cmd_preview_box.insert("1.0", cmd_str)
        self.cmd_preview_box.configure(state="disabled")

    def get_current_args(self):
        if not self.current_model: return None
        try:
            return {
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
                "threads": int(self.threads_input.get() or 4),
                "reference": self.ref_dropdown.get(),
                "ref_text": self.ref_text_input.get("1.0", "end-1c"),
                "speaker": self.speaker_dropdown.get(),
                "instruct": self.instruct_input.get("1.0", "end-1c")
            }
        except Exception: return None

    def generate(self):
        args = self.get_current_args()
        if not args or args["text"] == "...":
            self.status_label.configure(text=self.t("sentence"), text_color="red")
            return
        
        mtype = args.get("model_type", "base")
        if mtype == "design" and not args.get("instruct"):
            self.status_label.configure(text="Design model requires Instruct!", text_color="red")
            return

        self.report.unload_audio()
        self.status_label.configure(text=self.t("status_running"), text_color="white")
        self.generate_btn.configure(state="disabled")

        def run():
            try:
                result = self.engine.run_synthesis(args)
            except Exception as err:
                msg = str(err)
                self.after(0, lambda m=msg: self.status_label.configure(text=m[:80], text_color="red"))
                self.after(0, lambda: self.generate_btn.configure(state="normal"))
                return

            # Persist the PCM to the requested output path so the report
            # view can load it back via the existing audio player.
            try:
                out_path = args.get("output", "output/out.wav")
                abs_out = str(self.engine.save_audio(result, out_path))
            except Exception as err:
                msg = f"save failed: {err}"
                self.after(0, lambda m=msg: self.status_label.configure(text=m, text_color="red"))
                self.after(0, lambda: self.generate_btn.configure(state="normal"))
                return

            metrics = self.engine.format_metrics(result)
            backend_info = self.engine.get_backend_info()
            expected = backend_info.get("llama", {}).get("backend", "") if backend_info else ""

            self.after(0, lambda: self.status_label.configure(text=self.t("status_success"), text_color="green"))
            self.after(0, lambda m=metrics, p=abs_out, e=expected, s=args["text"]:
                       self.report.display(m, p, e, s))
            self.after(0, lambda m=metrics: self.header.check_backends(m))
            if self.header.auto_play_var.get():
                self.after(300, lambda: self.report.play_audio())
            self.after(0, lambda: self.generate_btn.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def on_platform_change(self, val):
        self.platform = val
        self.update_command_preview()

if __name__ == "__main__":
    app = LunaVoxTTS()
    app.mainloop()
