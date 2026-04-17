"""The library view — installed models, reference voices, and native runtimes.

Each model row exposes a Pull button that reuses the CLI's downloader
(``ModelDownloader.download_converted_model``) but drives it from a
modal dialog so the long-running output stays visible to the user.
The Synthesize view picks up newly installed models the next time it's
opened (views are recreated on tab switch by :class:`LunaVoxApp`).

Each runtime row (llama.cpp, onnxruntime) lets the user either:
  * pick a prebuilt backend from the libs.json catalog and download it, or
  * drop / browse to a local archive (for custom or pinned-version builds).

Drop is wired through ``tkinterdnd2`` if the package is installed; we
fall back to a click-to-browse interaction otherwise so the feature
still works on a stock environment.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from lunavox.build.lib_downloader import LIBS_CONFIG, install_local_archive, update_library
from lunavox.cli._config import ResolvedConfig
from lunavox.model import ModelDownloader, all_models
from lunavox.model.config import ModelSpec

from ..i18n import Translator
from ..lib_info import read_lib_metadata
from ..theme import (
    BG_CARD,
    CORNER_RADIUS,
    FONT_BODY,
    FONT_HEADING,
    FONT_TITLE,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    TEXT_MUTED,
)
from ..widgets import ConversionDialog

try:  # tkinterdnd2 is optional — graceful fallback to click-to-browse.
    from tkinterdnd2 import DND_FILES  # pyright: ignore[reportMissingImports]

    _DND_AVAILABLE = True
except Exception:  # noqa: BLE001
    DND_FILES = "DND_Files"  # placeholder so symbol exists
    _DND_AVAILABLE = False


_RUNTIMES = ("llama", "onnx")


class LibraryView(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, master: Any, config: ResolvedConfig, translator: Translator) -> None:
        super().__init__(master, fg_color="transparent")
        self._config = config
        self._t = translator
        # Per-model row widgets so a completing task can re-enable the
        # pull button without a full view rebuild.
        self._row_widgets: dict[str, dict[str, Any]] = {}
        self._runtime_rows: dict[str, dict[str, Any]] = {}
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self._t("lib.title"), font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_MD)
        )

        # --- Models section ---
        models_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        models_card.grid(row=1, column=0, sticky="ew", pady=(0, SPACE_MD))
        models_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(models_card, text=self._t("lib.models_section"), font=FONT_HEADING).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        specs = all_models()
        for i, spec in enumerate(specs, start=1):
            self._build_model_row(models_card, i, spec)

        if not specs:
            ctk.CTkLabel(
                models_card, text=self._t("lib.no_models"), text_color=TEXT_MUTED
            ).grid(row=1, column=0, columnspan=2, sticky="w", padx=SPACE_LG, pady=SPACE_MD)

        ctk.CTkLabel(models_card, text="").grid(
            row=len(specs) + 1, column=0, columnspan=2, pady=(0, SPACE_SM)
        )

        # --- Native runtimes section ---
        runtimes_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        runtimes_card.grid(row=2, column=0, sticky="ew", pady=(0, SPACE_MD))
        runtimes_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            runtimes_card, text=self._t("lib.runtimes_section"), font=FONT_HEADING
        ).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )
        ctk.CTkLabel(
            runtimes_card,
            text=self._t("lib.runtime_drop_hint"),
            font=FONT_BODY,
            text_color=TEXT_MUTED,
            wraplength=720,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=SPACE_LG, pady=(0, SPACE_SM))

        meta = read_lib_metadata(self._config.project_root)
        for i, lib_name in enumerate(_RUNTIMES, start=2):
            self._build_runtime_row(runtimes_card, i, lib_name, meta.get(lib_name))
        ctk.CTkLabel(runtimes_card, text="").grid(
            row=len(_RUNTIMES) + 2, column=0, columnspan=3, pady=(0, SPACE_SM)
        )

        # --- References section ---
        refs_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        refs_card.grid(row=3, column=0, sticky="ew")
        refs_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(refs_card, text=self._t("lib.references_section"), font=FONT_HEADING).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        ref_dir = self._config.project_root / "ref"
        if ref_dir.exists():
            files = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in {".wav", ".json"})
        else:
            files = []

        if not files:
            ctk.CTkLabel(refs_card, text=self._t("lib.no_references"), text_color=TEXT_MUTED).grid(
                row=1, column=0, sticky="w", padx=SPACE_LG, pady=(0, SPACE_LG)
            )
        else:
            for i, p in enumerate(files, start=1):
                ctk.CTkLabel(refs_card, text=p.name, font=FONT_BODY).grid(
                    row=i, column=0, sticky="w", padx=SPACE_LG, pady=2
                )
            ctk.CTkLabel(refs_card, text="").grid(row=len(files) + 1, column=0, pady=(0, SPACE_SM))

    # --- model rows ----------------------------------------------------

    def _build_model_row(self, parent: Any, row_index: int, spec: ModelSpec) -> None:
        info = ctk.CTkLabel(
            parent,
            text=spec.display_name,
            font=FONT_BODY,
            text_color=TEXT_MUTED,
            justify="left",
        )
        info.grid(row=row_index, column=0, sticky="w", padx=(SPACE_LG, SPACE_SM), pady=2)

        pull_btn = ctk.CTkButton(
            parent,
            text=self._t("lib.pull_btn"),
            width=90,
            command=lambda s=spec: self._start_pull(s),
        )
        pull_btn.grid(row=row_index, column=1, sticky="e", padx=(SPACE_SM, SPACE_LG), pady=2)

        self._row_widgets[spec.name] = {"pull": pull_btn}

    # --- runtime rows --------------------------------------------------

    def _build_runtime_row(
        self, parent: Any, row_index: int, lib_name: str, info: Any
    ) -> None:
        title = LIBS_CONFIG["libraries"].get(lib_name, {}).get("display_name", lib_name)
        if info is not None:
            status = self._t("lib.runtime_installed").format(
                label=info.label, version=info.version
            )
        else:
            status = self._t("lib.runtime_missing")
        label_text = f"{title}\n{status}"

        # Use a frame so we can attach DnD targets and visually indicate the drop zone.
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.grid(
            row=row_index, column=0, columnspan=3, sticky="ew", padx=SPACE_LG, pady=2
        )
        row_frame.grid_columnconfigure(0, weight=1)

        info_label = ctk.CTkLabel(
            row_frame,
            text=label_text,
            font=FONT_BODY,
            text_color=TEXT_MUTED,
            justify="left",
        )
        info_label.grid(row=0, column=0, sticky="w")

        download_btn = ctk.CTkButton(
            row_frame,
            text=self._t("lib.runtime_download"),
            width=110,
            command=lambda lib=lib_name: self._open_backend_picker(lib),
        )
        download_btn.grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

        browse_btn = ctk.CTkButton(
            row_frame,
            text=self._t("synth.browse"),
            width=90,
            fg_color="transparent",
            border_width=1,
            command=lambda lib=lib_name: self._browse_archive(lib),
        )
        browse_btn.grid(row=0, column=2, sticky="e", padx=(SPACE_SM, 0))

        self._runtime_rows[lib_name] = {
            "frame": row_frame,
            "info": info_label,
            "download": download_btn,
            "browse": browse_btn,
        }

        if _DND_AVAILABLE:
            try:
                row_frame.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                row_frame.dnd_bind(  # type: ignore[attr-defined]
                    "<<Drop>>",
                    lambda event, lib=lib_name: self._on_runtime_drop(lib, event),
                )
            except Exception:  # noqa: BLE001
                pass

    # --- runtime actions -----------------------------------------------

    def _set_runtime_busy(self, lib_name: str, busy: bool) -> None:
        widgets = self._runtime_rows.get(lib_name)
        if not widgets:
            return
        state = "disabled" if busy else "normal"
        widgets["download"].configure(state=state)
        widgets["browse"].configure(state=state)

    def _refresh_runtime_status(self, lib_name: str) -> None:
        widgets = self._runtime_rows.get(lib_name)
        if not widgets:
            return
        info = read_lib_metadata(self._config.project_root).get(lib_name)
        title = LIBS_CONFIG["libraries"].get(lib_name, {}).get("display_name", lib_name)
        if info is not None:
            status = self._t("lib.runtime_installed").format(
                label=info.label, version=info.version
            )
        else:
            status = self._t("lib.runtime_missing")
        widgets["info"].configure(text=f"{title}\n{status}")

    def _open_backend_picker(self, lib_name: str) -> None:
        backends = list(LIBS_CONFIG["libraries"].get(lib_name, {}).get("backends", {}).keys())
        if not backends:
            return
        _BackendPicker(
            self,
            translator=self._t,
            lib_name=lib_name,
            backends=backends,
            on_pick=lambda backend, lib=lib_name: self._start_runtime_download(lib, backend),
        )

    def _start_runtime_download(self, lib_name: str, backend: str) -> None:
        self._set_runtime_busy(lib_name, True)
        title = self._t("lib.runtime_install_title").format(lib=lib_name, backend=backend)
        project_root = self._config.project_root

        def target(log: Any) -> None:
            log(title)
            update_library(lib_name, backend, str(project_root))
            log(self._t("lib.runtime_install_done").format(lib=lib_name, backend=backend))

        def on_done(_success: bool) -> None:
            self._set_runtime_busy(lib_name, False)
            self._refresh_runtime_status(lib_name)

        ConversionDialog(
            self,
            translator=self._t,
            title=title,
            thread_target=target,
            on_done=on_done,
        )

    def _browse_archive(self, lib_name: str) -> None:
        path = filedialog.askopenfilename(
            title=self._t("lib.runtime_picker_title"),
            filetypes=[
                ("Archives", "*.zip *.nupkg *.tgz *.tar.gz"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._install_local(lib_name, Path(path))

    def _on_runtime_drop(self, lib_name: str, event: Any) -> None:
        # tkinterdnd2 packs paths as a tcl-style brace-delimited list.
        raw = str(getattr(event, "data", "")).strip()
        if not raw:
            return
        path = _parse_dnd_path(raw)
        if path is None:
            return
        self._install_local(lib_name, path)

    def _install_local(self, lib_name: str, archive: Path) -> None:
        self._set_runtime_busy(lib_name, True)
        title = self._t("lib.runtime_install_title").format(lib=lib_name, backend=archive.name)
        project_root = self._config.project_root

        def target(log: Any) -> None:
            log(self._t("lib.runtime_install_drop").format(lib=lib_name))
            log(str(archive))
            install_local_archive(
                lib_name, str(archive), str(project_root), label=archive.stem
            )
            log(self._t("lib.runtime_install_done").format(lib=lib_name, backend=archive.name))

        def on_done(_success: bool) -> None:
            self._set_runtime_busy(lib_name, False)
            self._refresh_runtime_status(lib_name)

        ConversionDialog(
            self,
            translator=self._t,
            title=title,
            thread_target=target,
            on_done=on_done,
        )

    # --- model task orchestration -------------------------------------

    def _set_row_busy(self, model_name: str, busy: bool) -> None:
        w = self._row_widgets.get(model_name)
        if not w:
            return
        w["pull"].configure(state="disabled" if busy else "normal")

    def _start_pull(self, spec: ModelSpec) -> None:
        self._set_row_busy(spec.name, True)
        title = self._t("lib.pulling_title").format(name=spec.name)
        project_root = self._config.project_root

        def target(log: Any) -> None:
            log(self._t("lib.task_pull_start").format(name=spec.name))
            dest_dir = project_root / "models" / spec.name

            total_files = _probe_total_files(spec.name)
            if total_files:
                log(
                    self._t("lib.task_pull_plan").format(
                        count=total_files, name=spec.name
                    )
                )

            stop = threading.Event()
            monitor = threading.Thread(
                target=_heartbeat,
                args=(dest_dir, total_files, stop, log, self._t),
                daemon=True,
            )
            monitor.start()
            try:
                ModelDownloader.download_converted_model(spec.name, project_root)
            finally:
                stop.set()

        def on_done(_success: bool) -> None:
            self._set_row_busy(spec.name, False)

        ConversionDialog(
            self,
            translator=self._t,
            title=title,
            thread_target=target,
            on_done=on_done,
        )


class _BackendPicker(ctk.CTkToplevel):  # pyright: ignore[reportUntypedBaseClass]
    """Tiny modal that lets the user pick which backend variant to download."""

    def __init__(
        self,
        master: Any,
        translator: Translator,
        lib_name: str,
        backends: list[str],
        on_pick: Any,
    ) -> None:
        super().__init__(master)
        self.title(translator("lib.runtime_picker_title"))
        self.geometry("360x320")
        self.transient(master)
        self.after(50, self._grab)
        self._on_pick = on_pick

        ctk.CTkLabel(
            self,
            text=f"{translator('lib.runtime_picker_title')} — {lib_name}",
            font=FONT_HEADING,
        ).pack(padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM), anchor="w")

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=SPACE_LG, pady=SPACE_SM)
        for backend in backends:
            ctk.CTkButton(
                scroll,
                text=backend,
                anchor="w",
                command=lambda b=backend: self._pick(b),
            ).pack(fill="x", pady=2)

    def _grab(self) -> None:
        try:
            self.grab_set()
        except Exception:  # noqa: BLE001
            pass

    def _pick(self, backend: str) -> None:
        try:
            self.grab_release()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()
        self._on_pick(backend)


def _parse_dnd_path(raw: str) -> Optional[Path]:
    """Extract one filesystem path from a tkdnd ``<<Drop>>`` payload.

    tkdnd packs paths as ``{C:/foo bar.zip} {D:/baz.zip}`` when names
    contain spaces; on a single-file drop without spaces the payload is
    just the bare path. Take the first entry either way.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{") and "}" in raw:
        first = raw[1 : raw.index("}")]
    else:
        first = raw.split()[0]
    p = Path(first)
    return p if p.exists() else None


def _probe_total_files(model_name: str) -> int:
    """Count files in the converted-artifacts repo for ``model_name``.

    Network best-effort — returns 0 on any failure so the heartbeat
    falls back to file-count-only reporting.
    """
    try:
        from huggingface_hub import list_repo_files  # pyright: ignore[reportUnknownVariableType]
    except Exception:  # noqa: BLE001
        return 0
    try:
        files = list_repo_files(repo_id="wkwong/Lunavox-Qwen3-TTS-GGUF")
    except Exception:  # noqa: BLE001
        return 0
    prefix = f"{model_name}/"
    return sum(1 for f in files if f.startswith(prefix))


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes = num_bytes // 1024 if unit == "B" else num_bytes / 1024  # type: ignore[assignment]
    return f"{num_bytes} B"


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


def _scan_dest(dest_dir: Path) -> tuple[int, int]:
    if not dest_dir.exists():
        return 0, 0
    files = 0
    size = 0
    try:
        for p in dest_dir.rglob("*"):
            try:
                if p.is_file():
                    files += 1
                    size += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return files, size


def _heartbeat(
    dest_dir: Path,
    total_files: int,
    stop: threading.Event,
    log: Any,
    translator: Translator,
) -> None:
    start = time.monotonic()
    # Small initial delay so the first tick has something to report
    # rather than "0 files · 0 B".
    if stop.wait(1.5):
        return
    while not stop.is_set():
        files, size = _scan_dest(dest_dir)
        elapsed = _fmt_elapsed(time.monotonic() - start)
        size_str = _fmt_size(size)
        if total_files:
            log(
                translator("lib.task_pull_progress_counted").format(
                    files=files, total=total_files, size=size_str, elapsed=elapsed
                )
            )
        else:
            log(
                translator("lib.task_pull_progress_simple").format(
                    files=files, size=size_str, elapsed=elapsed
                )
            )
        if stop.wait(2.0):
            return
