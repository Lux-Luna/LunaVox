"""Modal dialog that streams a long-running task's log into a textbox.

Two drive modes:

* ``cmd=[...]``  — spawn a subprocess with ``stdout=PIPE`` and pipe each
  line back onto the Tk main loop via :meth:`self.after`. Used for
  ``lunavox model convert``, whose pipeline is a sequence of subprocess
  calls and emits plenty of output to show.
* ``thread_target=fn`` — run an arbitrary callable on a background
  thread. The callable receives a ``log(msg: str)`` callback that
  marshals its argument back onto the Tk main loop. Used for Pull,
  where ``huggingface_hub.snapshot_download`` hides its own progress
  bar and we only need to report start/finish.

Exactly one of the two must be supplied. The modal swaps its Cancel
button for a Close button when the task finishes, and calls
``on_done(success)`` on the Tk loop so the caller can refresh badges
or re-enable buttons.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from typing import Any, Optional

import customtkinter as ctk  # pyright: ignore[reportMissingImports]

from ..i18n import Translator
from ..theme import (
    BG_CARD,
    CORNER_RADIUS,
    FONT_BODY,
    FONT_HEADING,
    FONT_MONO,
    SPACE_LG,
    SPACE_SM,
)

ThreadTarget = Callable[[Callable[[str], None]], None]


class ConversionDialog(ctk.CTkToplevel):  # pyright: ignore[reportUntypedBaseClass]
    """Modal that streams task output until the task completes."""

    def __init__(
        self,
        master: Any,
        translator: Translator,
        title: str,
        on_done: Callable[[bool], None],
        cmd: Optional[list[str]] = None,
        thread_target: Optional[ThreadTarget] = None,
    ) -> None:
        super().__init__(master)
        if (cmd is None) == (thread_target is None):
            raise ValueError("ConversionDialog requires exactly one of cmd / thread_target")
        self._t = translator
        self._on_done = on_done
        self._cmd = cmd
        self._thread_target = thread_target
        self._proc: Optional[subprocess.Popen[str]] = None
        self._done = False
        self._success = False

        self.title(title)
        self.geometry("760x440")
        self.minsize(560, 320)
        self.transient(master)
        # grab_set must be called after the window is viewable; schedule it.
        self.after(50, self._grab)
        self.protocol("WM_DELETE_WINDOW", self._on_close_clicked)

        self._build()
        self._start()

    # --- layout --------------------------------------------------------

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(self, text=self.title(), font=FONT_HEADING)
        header.grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM)
        )

        log_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=CORNER_RADIUS)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=SPACE_SM)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        self._log_box = ctk.CTkTextbox(
            log_frame, font=FONT_MONO, activate_scrollbars=True, wrap="none"
        )
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        self._log_box.configure(state="disabled")

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_SM, SPACE_LG))
        footer.grid_columnconfigure(0, weight=1)

        self._status = ctk.CTkLabel(footer, text="", font=FONT_BODY)
        self._status.grid(row=0, column=0, sticky="w")

        self._button = ctk.CTkButton(
            footer, text=self._t("lib.dialog_cancel"), command=self._on_cancel_clicked
        )
        self._button.grid(row=0, column=1, sticky="e")

    def _grab(self) -> None:
        try:
            self.grab_set()
        except Exception:  # noqa: BLE001 — grab_set races on some WMs
            pass

    # --- start ---------------------------------------------------------

    def _start(self) -> None:
        if self._cmd is not None:
            threading.Thread(target=self._run_subprocess, daemon=True).start()
        else:
            assert self._thread_target is not None
            threading.Thread(target=self._run_thread_target, daemon=True).start()

    def _run_subprocess(self) -> None:
        try:
            # bufsize=1 + text=True = line-buffered reads; stderr merged into
            # stdout so we don't need a second reader thread.
            self._proc = subprocess.Popen(  # noqa: S603 — cmd is built by us, not user
                self._cmd or [],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._append_log(line.rstrip("\n"))
            rc = self._proc.wait()
            ok = rc == 0
            if not ok:
                self._append_log(f"[exit code {rc}]")
            self._finish(ok)
        except Exception as err:  # noqa: BLE001
            self._append_log(self._t("lib.task_failed").format(err=err))
            self._finish(False)

    def _run_thread_target(self) -> None:
        try:
            assert self._thread_target is not None
            self._thread_target(self._append_log)
            self._finish(True)
        except Exception as err:  # noqa: BLE001
            self._append_log(self._t("lib.task_failed").format(err=err))
            self._finish(False)

    # --- log updates (thread-safe) ------------------------------------

    def _append_log(self, line: str) -> None:
        # Bounce onto the Tk loop so tkinter isn't touched from a thread.
        self.after(0, self._append_log_main, line)

    def _append_log_main(self, line: str) -> None:
        if self._done and line == "":
            return
        self._log_box.configure(state="normal")
        self._log_box.insert("end", line + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _finish(self, success: bool) -> None:
        self.after(0, self._finish_main, success)

    def _finish_main(self, success: bool) -> None:
        if self._done:
            return
        self._done = True
        self._success = success
        self._button.configure(text=self._t("lib.dialog_close"))
        status_key = "lib.task_done" if success else "lib.task_cancelled"
        self._status.configure(text=self._t(status_key))
        try:
            self._on_done(success)
        except Exception:  # noqa: BLE001 — don't let callbacks break the modal
            pass

    # --- button handling ----------------------------------------------

    def _on_cancel_clicked(self) -> None:
        if self._done:
            self._close()
            return
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        # Thread-target tasks can't be cancelled cooperatively here; mark
        # them as done so the dialog closes cleanly once the thread wraps.
        self._append_log(self._t("lib.task_cancelled"))
        self._finish(False)

    def _on_close_clicked(self) -> None:
        # The WM_DELETE_WINDOW handler — same semantics as Cancel.
        self._on_cancel_clicked()
        if self._done:
            self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()
