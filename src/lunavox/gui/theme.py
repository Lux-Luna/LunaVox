"""Visual tokens shared across every GUI view.

Kept as a flat module of constants rather than a style framework so
every widget can read the same numbers without a lookup layer. The
palette is a deep-purple Luna theme that matches the project brand
and reads in both light and dark modes (customtkinter honors system
appearance automatically when we pass the named colors).
"""

from __future__ import annotations

# Primary palette — the "Luna" deep purple plus a muted accent.
PRIMARY = "#6A5ACD"
PRIMARY_HOVER = "#7B68EE"
ACCENT = "#9F7AEA"
DANGER = "#E53E3E"
SUCCESS = "#38A169"

# Card / surface colors. customtkinter picks the first element for
# light mode and the second for dark mode.
BG_SURFACE = ("#F7F5FB", "#1A1625")
BG_CARD = ("#FFFFFF", "#221B33")
BG_SIDEBAR = ("#EDE8F7", "#16111F")
TEXT_MUTED = ("#4A5568", "#A0AEC0")

# Spacing scale — used verbatim as pad{x,y} so the whole UI shares
# one ruler. Mixing arbitrary values is the fastest way to end up
# with a patchy layout.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 16
SPACE_LG = 24

# Type scale.
FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_HEADING = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_MONO = ("Consolas", 11)

CORNER_RADIUS = 12

# Fixed window geometry — big enough to show synth + stats cards on a
# laptop without scrolling, small enough to fit on a 1080p second display.
WINDOW_WIDTH = 1320
WINDOW_HEIGHT = 720
SIDEBAR_WIDTH = 220
