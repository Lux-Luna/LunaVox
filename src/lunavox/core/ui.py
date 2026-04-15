from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "stage": "bold magenta",
    }
)

# `force_terminal` is intentionally NOT set: on Windows shells whose stdout
# code page cannot render Braille spinner characters (e.g. cp936 via
# git-bash), Rich's legacy fallback crashes with UnicodeEncodeError. Let
# Rich auto-detect the terminal so the build driver can gracefully degrade
# to plain text in non-TTY contexts. `safe_box` keeps panel borders using
# ASCII-safe characters even when the font lacks Unicode box glyphs.
console = Console(theme=THEME, safe_box=True)
