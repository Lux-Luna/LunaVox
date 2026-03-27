import customtkinter as ctk
import subprocess
import threading
import sys
import os


# Model options matching CLI source
MODEL_OPTIONS = [
    ("base_small", "Qwen3-TTS-12Hz-0.6B-Base"),
    ("custom_small", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ("base", "Qwen3-TTS-12Hz-1.7B-Base"),
    ("custom", "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    ("design", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
]

# Platform options matching libs.json
PLATFORM_OPTIONS = [
    ("win_cpu", "Windows (CPU Only)"),
    ("win_vulkan", "Windows (Universal GPU - DML/Vulkan)"),
    ("win_cuda12", "Windows (NVIDIA CUDA 12.x)"),
    ("win_cuda13", "Windows (NVIDIA CUDA 13.x)"),
    ("linux_cuda12", "Linux (NVIDIA CUDA 12.x)"),
    ("linux_cpu", "Linux (CPU Only)"),
    ("macos_arm64", "macOS (Apple Silicon / Metal)"),
]


class SetupPage(ctk.CTkFrame):
    def __init__(self, master, t_func, on_back, on_lang_change, on_refresh):
        super().__init__(master, fg_color="transparent")
        self.t = t_func
        self.on_back = on_back
        self.on_lang_change = on_lang_change
        self.on_refresh = on_refresh
        self._running = False

        self.grid_columnconfigure(0, weight=1)
        self.setup_ui()

    def setup_ui(self):
        # --- Scrollable content ---
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        row = 0

        # ============ 1. Language Section ============
        self.lang_section = ctk.CTkFrame(self.scroll)
        self.lang_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.lang_section.grid_columnconfigure(1, weight=1)

        self.lang_section_label = ctk.CTkLabel(
            self.lang_section, text=self.t("setup_lang_section"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.lang_section_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 5), sticky="w")

        self.lang_label = ctk.CTkLabel(self.lang_section, text=self.t("setup_lang_label"), font=ctk.CTkFont(size=12))
        self.lang_label.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        self.lang_dropdown = ctk.CTkOptionMenu(
            self.lang_section, values=["English", "中文"],
            command=self._on_lang_select, width=180
        )
        self.lang_dropdown.grid(row=1, column=1, padx=12, pady=(0, 10), sticky="w")
        row += 1

        # ============ 2. Model Download Section ============
        self.model_section = ctk.CTkFrame(self.scroll)
        self.model_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.model_section.grid_columnconfigure(1, weight=1)

        self.model_section_label = ctk.CTkLabel(
            self.model_section, text=self.t("setup_model_section"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.model_section_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 5), sticky="w")

        # China warning (hidden by default, shown when lang=zh)
        self.china_warning = ctk.CTkLabel(
            self.model_section, text=self.t("setup_china_warning"),
            font=ctk.CTkFont(size=11), text_color="#E8A838", wraplength=700
        )
        # Only show if warning text is non-empty
        if self.t("setup_china_warning"):
            self.china_warning.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 5), sticky="w")

        self.model_label = ctk.CTkLabel(self.model_section, text=self.t("setup_model_label"), font=ctk.CTkFont(size=12))
        self.model_label.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        model_display = [f"{display}" for _, display in MODEL_OPTIONS] + [self.t("setup_model_all")]
        self.model_dropdown = ctk.CTkOptionMenu(self.model_section, values=model_display, width=340)
        self.model_dropdown.grid(row=2, column=1, padx=12, pady=(0, 10), sticky="w")

        self.model_download_btn = ctk.CTkButton(
            self.model_section, text=self.t("setup_model_download"),
            width=160, height=34, fg_color="#2D8B57", hover_color="#1E6B3F",
            command=self._download_model
        )
        self.model_download_btn.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")
        row += 1

        # ============ 3. Library Download Section ============
        self.libs_section = ctk.CTkFrame(self.scroll)
        self.libs_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.libs_section.grid_columnconfigure(1, weight=1)

        self.libs_section_label = ctk.CTkLabel(
            self.libs_section, text=self.t("setup_libs_section"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.libs_section_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 5), sticky="w")

        self.libs_note = ctk.CTkLabel(
            self.libs_section, text=self.t("setup_libs_note"),
            font=ctk.CTkFont(size=11), text_color="#AAAAAA", wraplength=700
        )
        self.libs_note.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 5), sticky="w")

        self.libs_label = ctk.CTkLabel(self.libs_section, text=self.t("setup_libs_label"), font=ctk.CTkFont(size=12))
        self.libs_label.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        platform_display = [display for _, display in PLATFORM_OPTIONS]
        self.libs_dropdown = ctk.CTkOptionMenu(self.libs_section, values=platform_display, width=380)
        self.libs_dropdown.grid(row=2, column=1, padx=12, pady=(0, 10), sticky="w")

        self.libs_download_btn = ctk.CTkButton(
            self.libs_section, text=self.t("setup_libs_download"),
            width=160, height=34, fg_color="#2D8B57", hover_color="#1E6B3F",
            command=self._download_libs
        )
        self.libs_download_btn.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")
        row += 1

        # ============ 4. Build Section ============
        self.build_section = ctk.CTkFrame(self.scroll)
        self.build_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.build_section.grid_columnconfigure(0, weight=1)

        self.build_section_label = ctk.CTkLabel(
            self.build_section, text=self.t("setup_build_section"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.build_section_label.grid(row=0, column=0, padx=12, pady=(10, 5), sticky="w")

        self.build_note = ctk.CTkLabel(
            self.build_section, text=self.t("setup_build_note"),
            font=ctk.CTkFont(size=11), text_color="#AAAAAA"
        )
        self.build_note.grid(row=1, column=0, padx=12, pady=(0, 5), sticky="w")

        self.build_btn = ctk.CTkButton(
            self.build_section, text=self.t("setup_build_btn"),
            width=160, height=34, fg_color="#1F6AA5", hover_color="#144870",
            command=self._run_build
        )
        self.build_btn.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")
        row += 1

        # ============ 5. Console Output ============
        self.console_section = ctk.CTkFrame(self.scroll)
        self.console_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.console_section.grid_columnconfigure(0, weight=1)

        self.console_label = ctk.CTkLabel(
            self.console_section, text=self.t("setup_console_title"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.console_label.grid(row=0, column=0, padx=12, pady=(10, 5), sticky="w")

        self.status_label = ctk.CTkLabel(
            self.console_section, text=self.t("setup_status_idle"),
            font=ctk.CTkFont(size=12), text_color="#5BA0D0"
        )
        self.status_label.grid(row=0, column=0, padx=12, pady=(10, 5), sticky="e")

        self.console_box = ctk.CTkTextbox(
            self.console_section, height=200,
            font=ctk.CTkFont(family="Consolas", size=11),
            state="disabled"
        )
        self.console_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

    # --- Language ---
    def _on_lang_select(self, value):
        new_lang = "zh" if value == "中文" else "en"
        self.on_lang_change(new_lang)
        self.update_texts()

    def set_lang_dropdown(self, lang):
        """Sync the dropdown to the current app language."""
        self.lang_dropdown.set("中文" if lang == "zh" else "English")

    # --- Console helpers ---
    def _append_console(self, text):
        self.console_box.configure(state="normal")
        self.console_box.insert("end", text)
        self.console_box.see("end")
        self.console_box.configure(state="disabled")

    def _clear_console(self):
        self.console_box.configure(state="normal")
        self.console_box.delete("1.0", "end")
        self.console_box.configure(state="disabled")

    def _set_status(self, key, color="#5BA0D0"):
        self.status_label.configure(text=self.t(key), text_color=color)

    def _set_buttons_state(self, state):
        self.model_download_btn.configure(state=state)
        self.libs_download_btn.configure(state=state)
        self.build_btn.configure(state=state)

    # --- Run CLI command in background ---
    def _run_cli(self, cmd_args, status_key, on_done=None):
        if self._running:
            return
        self._running = True
        self._clear_console()
        self._set_status(status_key, "#E8A838")
        self._set_buttons_state("disabled")

        lunavox_exe = "lunavox"

        def worker():
            try:
                full_cmd = [lunavox_exe] + cmd_args
                self.after(0, lambda c=" ".join(full_cmd): self._append_console(f"$ {c}\n\n"))

                # Force UTF-8 on Windows
                proc_env = os.environ.copy()
                proc_env["PYTHONUTF8"] = "1"
                
                proc = subprocess.Popen(
                    full_cmd,
                    cwd=str(self._get_project_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=proc_env,
                    text=True,
                    encoding="utf-8"
                )

                for line in proc.stdout:
                    self.after(0, lambda l=line: self._append_console(l))

                proc.wait()

                if proc.returncode == 0:
                    self.after(0, lambda: self._set_status("setup_status_success", "#4CAF50"))
                    if on_done:
                        self.after(100, on_done)
                else:
                    self.after(0, lambda: self._set_status("setup_status_error", "#E74C3C"))
            except FileNotFoundError:
                self.after(0, lambda: self._append_console(
                    "\n[ERROR] 'lunavox' command not found.\n"
                    "Make sure lunavox is installed: pip install lunavox\n"
                ))
                self.after(0, lambda: self._set_status("setup_status_error", "#E74C3C"))
            except Exception as e:
                self.after(0, lambda err=str(e): self._append_console(f"\n[ERROR] {err}\n"))
                self.after(0, lambda: self._set_status("setup_status_error", "#E74C3C"))
            finally:
                self._running = False
                self.after(0, lambda: self._set_buttons_state("normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _get_project_root(self):
        """Get the LunaVox project root (parent of GUI dir)."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Actions ---
    def _download_model(self):
        selected = self.model_dropdown.get()
        all_label = self.t("setup_model_all")

        if selected == all_label:
            model_ids = [key for key, _ in MODEL_OPTIONS]
            self._run_cli_sequential_models(model_ids)
        else:
            # Find the model id by display name
            model_id = None
            for key, display in MODEL_OPTIONS:
                if display == selected:
                    model_id = key
                    break
            if model_id:
                self._run_cli(
                    ["--yes", "pull-model", "--model", model_id],
                    "setup_status_downloading",
                    on_done=self.on_refresh
                )

    def _run_cli_sequential_models(self, model_ids):
        """Download multiple models sequentially in one thread."""
        if self._running:
            return
        self._running = True
        self._clear_console()
        self._set_status("setup_status_downloading", "#E8A838")
        self._set_buttons_state("disabled")

        def worker():
            try:
                for i, mid in enumerate(model_ids):
                    full_cmd = ["lunavox", "--yes", "pull-model", "--model", mid]
                    cmd_str = " ".join(full_cmd)
                    header = f"\n--- [{i+1}/{len(model_ids)}] $ {cmd_str} ---\n"
                    self.after(0, lambda h=header: self._append_console(h))

                    # Force UTF-8 on Windows
                    proc_env = os.environ.copy()
                    proc_env["PYTHONUTF8"] = "1"
                    
                    proc = subprocess.Popen(
                        full_cmd,
                        cwd=str(self._get_project_root()),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=proc_env,
                        text=True,
                        encoding="utf-8"
                    )
                    for line in proc.stdout:
                        self.after(0, lambda l=line: self._append_console(l))
                    proc.wait()

                    if proc.returncode != 0:
                        self.after(0, lambda m=mid: self._append_console(f"\n[WARN] Model '{m}' download may have failed.\n"))

                self.after(0, lambda: self._set_status("setup_status_success", "#4CAF50"))
                self.after(100, self.on_refresh)
            except Exception as e:
                self.after(0, lambda err=str(e): self._append_console(f"\n[ERROR] {err}\n"))
                self.after(0, lambda: self._set_status("setup_status_error", "#E74C3C"))
            finally:
                self._running = False
                self.after(0, lambda: self._set_buttons_state("normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _download_libs(self):
        selected_display = self.libs_dropdown.get()
        platform_key = None
        for key, display in PLATFORM_OPTIONS:
            if display == selected_display:
                platform_key = key
                break
        if platform_key:
            self._run_cli(
                ["--yes", "download-libs", "--platform", platform_key],
                "setup_status_downloading",
                on_done=self.on_refresh
            )

    def _run_build(self):
        self._run_cli(
            ["--yes", "build", "--clean"],
            "setup_status_building",
            on_done=self.on_refresh
        )

    # --- Update texts ---
    def update_texts(self):
        self.lang_section_label.configure(text=self.t("setup_lang_section"))
        self.lang_label.configure(text=self.t("setup_lang_label"))
        self.model_section_label.configure(text=self.t("setup_model_section"))
        self.model_label.configure(text=self.t("setup_model_label"))
        self.model_download_btn.configure(text=self.t("setup_model_download"))
        self.libs_section_label.configure(text=self.t("setup_libs_section"))
        self.libs_label.configure(text=self.t("setup_libs_label"))
        self.libs_note.configure(text=self.t("setup_libs_note"))
        self.libs_download_btn.configure(text=self.t("setup_libs_download"))
        self.build_section_label.configure(text=self.t("setup_build_section"))
        self.build_btn.configure(text=self.t("setup_build_btn"))
        self.build_note.configure(text=self.t("setup_build_note"))
        self.console_label.configure(text=self.t("setup_console_title"))
        self.status_label.configure(text=self.t("setup_status_idle"))

        # Update "ALL MODELS" label in dropdown
        model_display = [display for _, display in MODEL_OPTIONS] + [self.t("setup_model_all")]
        self.model_dropdown.configure(values=model_display)

        # China warning: show/hide based on whether the text is non-empty
        warning_text = self.t("setup_china_warning")
        if warning_text:
            self.china_warning.configure(text=warning_text)
            self.china_warning.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 5), sticky="w")
        else:
            self.china_warning.grid_forget()
