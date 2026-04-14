"""Setup page — in-process version.

Previously this spawned ``lunavox`` as a subprocess for every action and
streamed its stdout into a Tk textbox. That duplicated the CLI binding
and forced users to install the CLI on their PATH before the GUI could
do anything. We now import the same pure-Python entry points the CLI
uses (``ModelDownloader.download_converted_model``,
``download_platform_libs``, ``run_build``) and run them on a worker
thread with stdout/stderr redirected into a line-buffered pipe so the
console panel still shows live progress.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import traceback

import customtkinter as ctk

from lunavox.build.lib_downloader import download_platform_libs
from lunavox.build.main import run_build
from lunavox.core.project import resolve_project_root
from lunavox.model import ModelDownloader, all_models, model_keys


# Platform options matching libs.json. Kept here because the GUI catalog
# order is user-facing and differs from the raw libs.json dict order.
PLATFORM_OPTIONS = [
    ("win_cpu", "Windows (CPU Only)"),
    ("win_vulkan", "Windows (Universal GPU - DML/Vulkan)"),
    ("win_cuda12", "Windows (NVIDIA CUDA 12.x)"),
    ("win_cuda13", "Windows (NVIDIA CUDA 13.x)"),
    ("linux_cuda12", "Linux (NVIDIA CUDA 12.x)"),
    ("linux_cpu", "Linux (CPU Only)"),
    ("macos_arm64", "macOS (Apple Silicon / Metal)"),
]


class _LineTee(io.TextIOBase):
    """File-like object that buffers writes and flushes them by line.

    Every completed line is forwarded through ``on_line`` (expected to
    schedule a ``Tk.after(0, ...)`` call into the console textbox). We
    keep the behaviour narrow on purpose: no stream isatty(), no rich
    terminal handshake — just plain text with newlines.
    """

    def __init__(self, on_line):
        super().__init__()
        self._on_line = on_line
        self._buf: list[str] = []

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:  # type: ignore[override]
        if not text:
            return 0
        self._buf.append(text)
        if "\n" in text:
            joined = "".join(self._buf)
            lines = joined.splitlines(keepends=True)
            tail: list[str] = []
            for line in lines:
                if line.endswith("\n"):
                    self._on_line(line)
                else:
                    tail.append(line)
            self._buf = tail
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._on_line("".join(self._buf))
            self._buf = []


class SetupPage(ctk.CTkFrame):
    def __init__(self, master, t_func, on_back, on_lang_change, on_refresh):
        super().__init__(master, fg_color="transparent")
        self.t = t_func
        self.on_back = on_back
        self.on_lang_change = on_lang_change
        self.on_refresh = on_refresh
        self._running = False
        # Cache the catalog so the dropdown is stable across language
        # toggles and we don't re-query the HF cache on every rebuild.
        self._catalog = all_models()

        self.grid_columnconfigure(0, weight=1)
        self.setup_ui()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def setup_ui(self):
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        row = 0

        # ---- Language Section ----
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

        # ---- Model Download Section ----
        self.model_section = ctk.CTkFrame(self.scroll)
        self.model_section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        self.model_section.grid_columnconfigure(1, weight=1)

        self.model_section_label = ctk.CTkLabel(
            self.model_section, text=self.t("setup_model_section"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.model_section_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 5), sticky="w")

        self.china_warning = ctk.CTkLabel(
            self.model_section, text=self.t("setup_china_warning"),
            font=ctk.CTkFont(size=11), text_color="#E8A838", wraplength=700
        )
        if self.t("setup_china_warning"):
            self.china_warning.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 5), sticky="w")

        self.model_label = ctk.CTkLabel(self.model_section, text=self.t("setup_model_label"), font=ctk.CTkFont(size=12))
        self.model_label.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        model_display = [spec.display_name for spec in self._catalog] + [self.t("setup_model_all")]
        self.model_dropdown = ctk.CTkOptionMenu(self.model_section, values=model_display, width=340)
        self.model_dropdown.grid(row=2, column=1, padx=12, pady=(0, 10), sticky="w")

        self.model_download_btn = ctk.CTkButton(
            self.model_section, text=self.t("setup_model_download"),
            width=160, height=34, fg_color="#2D8B57", hover_color="#1E6B3F",
            command=self._download_model
        )
        self.model_download_btn.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")
        row += 1

        # ---- Library Download Section ----
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

        # ---- Build Section ----
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

        # ---- Console Output ----
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

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------
    def _on_lang_select(self, value):
        new_lang = "zh" if value == "中文" else "en"
        self.on_lang_change(new_lang)
        self.update_texts()

    def set_lang_dropdown(self, lang):
        self.lang_dropdown.set("中文" if lang == "zh" else "English")

    def _append_console(self, text: str):
        self.console_box.configure(state="normal")
        self.console_box.insert("end", text)
        self.console_box.see("end")
        self.console_box.configure(state="disabled")

    def _clear_console(self):
        self.console_box.configure(state="normal")
        self.console_box.delete("1.0", "end")
        self.console_box.configure(state="disabled")

    def _set_status(self, key: str, color: str = "#5BA0D0"):
        self.status_label.configure(text=self.t(key), text_color=color)

    def _set_buttons_state(self, state: str):
        self.model_download_btn.configure(state=state)
        self.libs_download_btn.configure(state=state)
        self.build_btn.configure(state=state)

    def _get_project_root(self):
        return resolve_project_root()

    # -----------------------------------------------------------------
    # Worker dispatch
    # -----------------------------------------------------------------
    def _run_task(self, title: str, status_key: str, fn, on_done=None):
        """Run ``fn`` on a background thread with stdout/stderr teed to
        the console panel.

        ``fn`` is called with no arguments. It must be self-contained —
        all inputs should be captured in a closure by the caller. Any
        exception is caught, formatted, and surfaced in the console.
        """
        if self._running:
            return
        self._running = True
        self._clear_console()
        self._set_status(status_key, "#E8A838")
        self._set_buttons_state("disabled")
        self._append_console(f"$ {title}\n\n")

        def push_line(line: str):
            self.after(0, lambda l=line: self._append_console(l))

        tee = _LineTee(push_line)

        def worker():
            # Force UTF-8 on Windows so HF Hub progress bars don't
            # explode on the default cp936 encoding.
            prev_env = os.environ.get("PYTHONUTF8")
            os.environ["PYTHONUTF8"] = "1"
            ok = False
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    fn()
                    tee.flush()
                ok = True
            except Exception:
                tee.flush()
                self.after(0, lambda tb=traceback.format_exc():
                           self._append_console(f"\n[ERROR]\n{tb}\n"))
            finally:
                if prev_env is None:
                    os.environ.pop("PYTHONUTF8", None)
                else:
                    os.environ["PYTHONUTF8"] = prev_env
                self._running = False
                if ok:
                    self.after(0, lambda: self._set_status("setup_status_success", "#4CAF50"))
                    if on_done:
                        self.after(100, on_done)
                else:
                    self.after(0, lambda: self._set_status("setup_status_error", "#E74C3C"))
                self.after(0, lambda: self._set_buttons_state("normal"))

        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _selected_model_keys(self) -> list[str]:
        selected = self.model_dropdown.get()
        if selected == self.t("setup_model_all"):
            return model_keys()
        for spec in self._catalog:
            if spec.display_name == selected:
                return [spec.name]
        return []

    def _download_model(self):
        keys = self._selected_model_keys()
        if not keys:
            return
        root = self._get_project_root()

        def work():
            for i, key in enumerate(keys, 1):
                print(f"--- [{i}/{len(keys)}] Pulling model: {key} ---")
                ModelDownloader.download_converted_model(key, root)

        title = "pull-model " + (",".join(keys) if len(keys) > 1 else keys[0])
        self._run_task(title, "setup_status_downloading", work, on_done=self.on_refresh)

    def _download_libs(self):
        selected_display = self.libs_dropdown.get()
        platform_key = None
        for key, display in PLATFORM_OPTIONS:
            if display == selected_display:
                platform_key = key
                break
        if not platform_key:
            return
        root = self._get_project_root()

        def work():
            print(f"--- Downloading libs for platform: {platform_key} ---")
            download_platform_libs(platform_key, str(root))

        self._run_task(
            f"download-libs --platform {platform_key}",
            "setup_status_downloading",
            work,
            on_done=self.on_refresh,
        )

    def _run_build(self):
        root = self._get_project_root()

        def work():
            print("--- Building LunaVox C++ engine (clean) ---")
            run_build(
                root=root,
                clean=True,
                jobs=4,
                toolchain="auto",
                verbose=False,
            )

        self._run_task("build --clean", "setup_status_building", work, on_done=self.on_refresh)

    # -----------------------------------------------------------------
    # Text updates (language toggle)
    # -----------------------------------------------------------------
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

        model_display = [spec.display_name for spec in self._catalog] + [self.t("setup_model_all")]
        self.model_dropdown.configure(values=model_display)

        warning_text = self.t("setup_china_warning")
        if warning_text:
            self.china_warning.configure(text=warning_text)
            self.china_warning.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 5), sticky="w")
        else:
            self.china_warning.grid_forget()
