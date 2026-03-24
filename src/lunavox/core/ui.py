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

console = Console(theme=THEME, force_terminal=True, safe_box=True)

