import customtkinter as ctk

class HeaderFrame(ctk.CTkFrame):
    def __init__(self, master, t_func, on_lang_toggle, on_platform_change):
        super().__init__(master)
        self.t = t_func
        self.on_lang_toggle = on_lang_toggle
        self.on_platform_change = on_platform_change

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        
        # Left side: Backend Status
        self.info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.info_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.onnx_label = ctk.CTkLabel(self.info_frame, text="ONNX Runtime: --", font=ctk.CTkFont(size=12, weight="bold"), text_color="white")
        self.onnx_label.pack(side="left", padx=5)
        self.llama_label = ctk.CTkLabel(self.info_frame, text="Llama.cpp: --", font=ctk.CTkFont(size=12, weight="bold"), text_color="white")
        self.llama_label.pack(side="left", padx=5)

        # Right side: Settings Group
        self.settings_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.settings_frame.grid(row=0, column=1, padx=10, pady=5, sticky="e")

        self.auto_play_var = ctk.BooleanVar(value=True)
        self.auto_play_check = ctk.CTkCheckBox(self.settings_frame, text="Play", variable=self.auto_play_var, width=60, font=ctk.CTkFont(size=11))
        self.auto_play_check.pack(side="left", padx=5)

        self.platform_dropdown = ctk.CTkOptionMenu(self.settings_frame, values=["Windows", "macOS", "Linux"], command=self.on_platform_change, width=100, font=ctk.CTkFont(size=11), height=24)
        self.platform_dropdown.pack(side="left", padx=5)
        self.platform_dropdown.set("Windows")

        self.lang_btn = ctk.CTkButton(self.settings_frame, text="中文", width=60, height=24, font=ctk.CTkFont(size=11), command=self.on_lang_toggle)
        self.lang_btn.pack(side="left", padx=5)

    def update_info(self, backend_info):
        """Show expected backends from metadata.json (initial state)."""
        self.expected_info = backend_info
        if backend_info:
            onnx = backend_info.get("onnx", {}).get("provider", "CPU")
            llama = backend_info.get("llama", {}).get("backend", "cpu")
            self.onnx_label.configure(text=f"ONNX Runtime: {onnx}")
            self.llama_label.configure(text=f"Llama.cpp: {llama}")

    def check_backends(self, metrics):
        """Compare actual backends from synthesis output against expected metadata.
        Turn labels yellow with warning text on mismatch."""
        if not hasattr(self, 'expected_info') or not self.expected_info:
            return

        # Llama backend check
        expected_llama = self.expected_info.get("llama", {}).get("backend", "").lower()
        actual_llama = metrics.get("actual_backend_llama", "").lower()
        if expected_llama and actual_llama and actual_llama != "n/a":
            if expected_llama not in actual_llama:
                self.llama_label.configure(
                    text=f"⚠ Llama.cpp: {metrics.get('actual_backend_llama', 'N/A')}",
                    text_color="#FFD700"
                )
            else:
                self.llama_label.configure(
                    text=f"Llama.cpp: {metrics.get('actual_backend_llama', 'N/A')}",
                    text_color="#4CAF50"
                )

        # ONNX backend check
        expected_onnx = self.expected_info.get("onnx", {}).get("provider", "").lower()
        actual_onnx = metrics.get("actual_backend_onnx", "").lower()
        if expected_onnx and actual_onnx and actual_onnx != "n/a":
            if expected_onnx not in actual_onnx:
                self.onnx_label.configure(
                    text=f"⚠ ONNX Runtime: {metrics.get('actual_backend_onnx', 'N/A')}",
                    text_color="#FFD700"
                )
            else:
                self.onnx_label.configure(
                    text=f"ONNX Runtime: {metrics.get('actual_backend_onnx', 'N/A')}",
                    text_color="#4CAF50"
                )

    def update_texts(self):
        self.lang_btn.configure(text=self.t("language_switch"))
        self.auto_play_check.configure(text=self.t("auto_play_short"))
