import customtkinter as ctk
import os
import shutil
import threading

class ReportFrame(ctk.CTkFrame):
    def __init__(self, master, t_func):
        super().__init__(master)
        self.t = t_func
        self.current_audio = None
        self.is_playing = False
        self.playback_thread = None
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.title_label = ctk.CTkLabel(self, text="Report", font=ctk.CTkFont(weight="bold", size=14))
        self.title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")

        # Row 1-2: Primary Stats
        self.rtf_val = self.create_metric(1, 0, "rtf")
        self.latency_val = self.create_metric(1, 1, "latency")
        self.ram_val = self.create_metric(1, 2, "ram")
        self.vram_val = self.create_metric(1, 3, "vram")

        # Row 3-4: Breakdown
        self.tok_val = self.create_metric(3, 0, "tokenization")
        self.spk_val = self.create_metric(3, 1, "speaker_encoding")
        self.code_val = self.create_metric(3, 2, "code_generation")
        self.audio_val = self.create_metric(3, 3, "audio_decoding")

        # Row 5-6: Backends
        self.llama_val = self.create_metric(5, 0, "actual_backend")
        self.onnx_val = self.create_metric(5, 2, "actual_backend_onnx")
        
        # Audio Player Bar (hidden initially)
        self.audio_bar = ctk.CTkFrame(self, fg_color="#1A1A2E", corner_radius=8)
        self.audio_bar.grid_columnconfigure(1, weight=1)
        # Will be shown via grid when audio is ready

        self.play_btn = ctk.CTkButton(
            self.audio_bar, text="▶", width=36, height=36,
            corner_radius=18, fg_color="#1F6AA5", hover_color="#144870",
            font=ctk.CTkFont(size=14), command=self.toggle_playback
        )
        self.play_btn.grid(row=0, column=0, padx=(10, 5), pady=8)

        self.audio_name_label = ctk.CTkLabel(
            self.audio_bar, text="", font=ctk.CTkFont(size=12),
            text_color="#CCCCCC", anchor="w"
        )
        self.audio_name_label.grid(row=0, column=1, padx=5, pady=8, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.audio_bar, height=4, progress_color="#1F6AA5")
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 8), sticky="ew")
        self.progress_bar.set(0)

        self.download_btn = ctk.CTkButton(
            self.audio_bar, text="⬇", width=36, height=36,
            corner_radius=18, fg_color="#2D8B57", hover_color="#1E6B3F",
            font=ctk.CTkFont(size=14), command=self.download_audio
        )
        self.download_btn.grid(row=0, column=2, padx=(5, 10), pady=8)

        self.warning_label = ctk.CTkLabel(self, text="", text_color="#E74C3C", font=ctk.CTkFont(size=12, weight="bold"))
        self.warning_label.grid(row=9, column=0, columnspan=4, padx=10, pady=5)

        # Store the target sentence for download naming
        self.target_sentence = ""

    def create_metric(self, r, c, key):
        lbl = ctk.CTkLabel(self, text=self.t(key), font=ctk.CTkFont(size=11), text_color="#AAAAAA")
        lbl.grid(row=r, column=c, padx=10, pady=(5, 0))
        val = ctk.CTkLabel(self, text="--", font=ctk.CTkFont(weight="bold", size=13))
        val.grid(row=r+1, column=c, padx=10, pady=(0, 5))
        return val

    def unload_audio(self):
        self.is_playing = False
        self.play_btn.configure(text="▶")
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception:
            pass

    def toggle_playback(self):
        if not self.current_audio:
            return
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.play_btn.configure(text="▶")
            self.progress_bar.set(0)
        else:
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            pygame.mixer.music.load(self.current_audio)
            pygame.mixer.music.play()
            self.is_playing = True
            self.play_btn.configure(text="⏸")
            # Start progress tracking
            self._track_progress()

    def _track_progress(self):
        """Track playback progress and update the bar."""
        import pygame
        if self.is_playing and pygame.mixer.music.get_busy():
            # Simple indeterminate-style progress (pulse)
            try:
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms > 0:
                    # Use a cycling progress for visual feedback (no duration info from pygame)
                    cycle = (pos_ms % 5000) / 5000.0
                    self.progress_bar.set(cycle)
            except Exception:
                pass
            self.after(100, self._track_progress)
        else:
            if self.is_playing:
                # Playback ended naturally
                self.is_playing = False
                self.play_btn.configure(text="▶")
                self.progress_bar.set(1.0)

    def play_audio(self):
        """Auto-play convenience method (called by main on synthesis success)."""
        if self.current_audio and not self.is_playing:
            self.toggle_playback()

    def download_audio(self):
        if not self.current_audio or not os.path.exists(self.current_audio):
            return
        # Build a safe filename from the target sentence
        safe_name = self.target_sentence.strip()
        if not safe_name:
            safe_name = "output"
        # Sanitize: remove chars illegal in filenames
        for ch in '<>:"/\\|?*\n\r':
            safe_name = safe_name.replace(ch, '')
        safe_name = safe_name[:80].strip()  # Limit length
        if not safe_name:
            safe_name = "output"
        
        dest = ctk.filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")],
            initialfile=f"{safe_name}.wav"
        )
        if dest:
            try:
                shutil.copy2(self.current_audio, dest)
            except Exception as e:
                print(f"Download failed: {e}")

    def display(self, metrics, audio_path=None, expected_backend=None, sentence=""):
        self.current_audio = audio_path
        self.target_sentence = sentence
        
        # Update values
        mapping = {
            self.latency_val: "latency",
            self.rtf_val: "rtf",
            self.ram_val: "ram",
            self.vram_val: "vram",
            self.tok_val: "tokenization",
            self.spk_val: "speaker_encoding",
            self.code_val: "code_generation",
            self.audio_val: "audio_decoding",
            self.llama_val: "actual_backend_llama",
            self.onnx_val: "actual_backend_onnx"
        }
        
        for widget, key in mapping.items():
            widget.configure(text=metrics.get(key, "N/A"))

        # Show audio bar if we have audio
        if audio_path and os.path.exists(audio_path):
            basename = os.path.basename(audio_path)
            self.audio_name_label.configure(text=basename)
            self.audio_bar.grid(row=7, column=0, columnspan=4, padx=10, pady=(5, 10), sticky="ew")
            self.progress_bar.set(0)
            self.play_btn.configure(text="▶")
            self.is_playing = False

        # Backend Warning Logic
        actual_llama = metrics.get("actual_backend_llama", "").lower()
        if expected_backend:
            exp = expected_backend.lower()
            if exp and exp not in actual_llama:
                self.warning_label.configure(text=f"{self.t('warning_backend')} ({expected_backend} -> {metrics.get('actual_backend_llama')})")
            else:
                self.warning_label.configure(text="")
        else:
            self.warning_label.configure(text="")

    def update_texts(self):
        self.title_label.configure(text=self.t("report_header"))
