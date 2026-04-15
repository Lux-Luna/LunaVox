"""Host-platform helpers.

All ``sys.platform`` / ``platform.system()`` branches in the Python side
live here. The rule mirrors the C++ side: other modules must not probe the
OS directly — they import from here so the selection logic stays in one
file and future platforms only require one edit.

The ``build.factory`` / ``build.get_*_class`` entry points are the single
allowed exception because they bind a host tag to an entire Builder class
rather than to a scalar.
"""

from __future__ import annotations

import sys


def shared_lib_name(stem: str) -> str:
    """Return the platform-appropriate shared-library filename for ``stem``.

    Windows: ``{stem}.dll``; macOS: ``lib{stem}.dylib``; Linux / other POSIX:
    ``lib{stem}.so``.
    """
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def executable_suffix() -> str:
    """Empty on POSIX, ``.exe`` on Windows. Useful when resolving binary
    paths like ``build/lunavox-cli``."""
    return ".exe" if sys.platform == "win32" else ""


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")
