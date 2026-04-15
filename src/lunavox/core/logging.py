"""Unified process-wide logger for the Python side.

Every writer — CLI session header, build driver stage output, conversion
pipeline subprocess capture, GUI-bridged C++ log callback — appends to the
same ``logs/latest.log`` through this module. A single ``threading.Lock``
serializes writes so concurrent tasks (e.g. Rich status spinner + build
subprocess + model download) cannot corrupt each other.

Design notes:

- The log file is a plain append stream. No level filtering and no
  handler chain; the consumer decides what to write, we just persist.
- ``session_start(path)`` truncates and writes a session header. The CLI
  calls this once per invocation; nothing else should.
- Writers call ``append(text)``. The module is a no-op until
  ``session_start`` has been called, so importing the logger from a
  library module is safe even in contexts where no log file exists
  (tests, sdist builds, IDE completion).
- Keep this file dependency-free: no Rich, no lunavox imports. The UI
  console lives in ``core.ui`` and is separate on purpose — users can
  silence the console without losing the log, and vice versa.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_log_path: Optional[Path] = None


def session_start(path: Path, header: str = "") -> None:
    """Truncate ``path`` and register it as the process-wide log file.

    Subsequent :func:`append` calls will go here. Safe to call multiple
    times per process — the last call wins (CLI does this once at
    startup; tests may rebind to a tmp path).
    """
    global _log_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _log_path = path
        with open(path, "w", encoding="utf-8") as f:
            if header:
                f.write(header)
                if not header.endswith("\n"):
                    f.write("\n")


def append(text: str) -> None:
    """Append ``text`` to the active session log. No-op if unbound."""
    if _log_path is None:
        return
    with _lock:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def current_path() -> Optional[Path]:
    return _log_path
