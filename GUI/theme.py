# LunaVox GUI Theme System
# Supports dark and light themes using native tkinter/ttk only

from tkinter import ttk

# =============================================================================
# Color Palettes
# =============================================================================

DARK_COLORS = {
    # Backgrounds
    "bg_dark": "#1a1b26",        # Main window background
    "bg_surface": "#24283b",     # Card/panel background
    "bg_elevated": "#2a2e42",    # Elevated elements (hover states)
    "bg_input": "#1f2335",       # Input field background
    
    # Primary accent
    "primary": "#7aa2f7",        # Main accent (blue)
    "primary_hover": "#89b4fa",  # Hover state
    "primary_pressed": "#5d7cc7", # Pressed state
    
    # Secondary
    "secondary": "#bb9af7",      # Purple accent
    
    # Text
    "text_primary": "#c0caf5",   # Main text
    "text_secondary": "#9aa5ce", # Secondary text
    "text_muted": "#565f89",     # Muted/disabled text
    "text_inverse": "#1a1b26",   # Text on primary buttons
    
    # Semantic
    "success": "#9ece6a",        # Green
    "warning": "#e0af68",        # Orange
    "error": "#f7768e",          # Red
    "info": "#7dcfff",           # Cyan
    
    # Borders
    "border_default": "#3b4261",
    "border_focus": "#7aa2f7",
}

LIGHT_COLORS = {
    # Backgrounds
    "bg_dark": "#f8f9fa",        # Main window background (light gray)
    "bg_surface": "#ffffff",     # Card/panel background (white)
    "bg_elevated": "#e9ecef",    # Elevated elements (hover states)
    "bg_input": "#ffffff",       # Input field background
    
    # Primary accent
    "primary": "#3b82f6",        # Main accent (blue)
    "primary_hover": "#60a5fa",  # Hover state
    "primary_pressed": "#2563eb", # Pressed state
    
    # Secondary
    "secondary": "#8b5cf6",      # Purple accent
    
    # Text
    "text_primary": "#1f2937",   # Main text (dark gray)
    "text_secondary": "#4b5563", # Secondary text
    "text_muted": "#9ca3af",     # Muted/disabled text
    "text_inverse": "#ffffff",   # Text on primary buttons
    
    # Semantic
    "success": "#22c55e",        # Green
    "warning": "#f59e0b",        # Orange
    "error": "#ef4444",          # Red
    "info": "#06b6d4",           # Cyan
    
    # Borders
    "border_default": "#d1d5db",
    "border_focus": "#3b82f6",
}

# =============================================================================
# Current Theme State
# =============================================================================

# Default to dark theme
_current_theme = "dark"
COLORS = DARK_COLORS.copy()

def get_current_theme():
    """Get the current theme name."""
    return _current_theme

def set_theme(theme_name: str):
    """Set the current theme ('dark' or 'light')."""
    global _current_theme, COLORS
    _current_theme = theme_name
    if theme_name == "light":
        COLORS.update(LIGHT_COLORS)
    else:
        COLORS.update(DARK_COLORS)

# =============================================================================
# Typography
# =============================================================================

FONTS = {
    "family":  "Segoe UI" if __import__('sys').platform == 'win32' else "SF Pro Display",
    "family_mono": "Consolas" if __import__('sys').platform == 'win32' else "SF Mono",
    "size_xs": 9,
    "size_sm": 10,
    "size_base": 11,
    "size_lg": 13,
    "size_xl": 16,
    "size_2xl": 20,
}

# =============================================================================
# Spacing
# =============================================================================

SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "2xl": 32,
}

# =============================================================================
# Apply Theme to ttk Style
# =============================================================================

def apply_theme(root, theme_name: str = None):
    """Apply the theme to the entire application.
    
    Args:
        root: The root Tk window
        theme_name: 'dark' or 'light'. If None, uses current theme.
    """
    if theme_name:
        set_theme(theme_name)
    
    style = ttk.Style(root)
    
    # Use clam as base theme for best customization
    style.theme_use('clam')
    
    # --- General ---
    root.configure(bg=COLORS["bg_dark"])
    
    # --- TFrame ---
    style.configure("TFrame", background=COLORS["bg_dark"])
    style.configure("Card.TFrame", background=COLORS["bg_surface"])
    
    # --- TLabel ---
    style.configure("TLabel", 
                    background=COLORS["bg_dark"], 
                    foreground=COLORS["text_primary"],
                    font=(FONTS["family"], FONTS["size_base"]))
    
    style.configure("Card.TLabel", 
                    background=COLORS["bg_surface"], 
                    foreground=COLORS["text_primary"])
    
    style.configure("Header.TLabel", 
                    font=(FONTS["family"], FONTS["size_lg"], "bold"),
                    foreground=COLORS["text_primary"])
    
    style.configure("Title.TLabel", 
                    font=(FONTS["family"], FONTS["size_xl"], "bold"),
                    foreground=COLORS["primary"])
    
    style.configure("Muted.TLabel", 
                    foreground=COLORS["text_muted"])
    
    # --- TButton ---
    style.configure("TButton",
                    font=(FONTS["family"], FONTS["size_base"]),
                    padding=(SPACING["md"], SPACING["sm"]),
                    background=COLORS["bg_elevated"],
                    foreground=COLORS["text_primary"],
                    borderwidth=0)
    style.map("TButton",
              background=[('active', COLORS["primary"]), ('pressed', COLORS["bg_input"]), ('disabled', COLORS["bg_surface"])],
              foreground=[('active', COLORS["text_inverse"]), ('disabled', COLORS["text_muted"])])
    
    style.configure("Primary.TButton",
                    background=COLORS["primary"],
                    foreground=COLORS["text_inverse"],
                    font=(FONTS["family"], FONTS["size_base"], "bold"),
                    borderwidth=0)
    style.map("Primary.TButton",
              background=[('active', COLORS["primary_hover"]), ('pressed', COLORS["primary_pressed"]), ('disabled', COLORS["bg_surface"])],
              foreground=[('disabled', COLORS["text_muted"])])
    
    style.configure("Success.TButton",
                    background=COLORS["success"],
                    foreground=COLORS["text_inverse"],
                    borderwidth=0)
    style.map("Success.TButton",
              background=[('active', '#a8d475'), ('pressed', '#8bc55a')])
    
    style.configure("Danger.TButton",
                    background=COLORS["error"],
                    foreground=COLORS["text_inverse"],
                    borderwidth=0)
    style.map("Danger.TButton",
              background=[('active', '#ff8fa0'), ('pressed', '#e5657a')])

    
    # --- TEntry ---
    style.configure("TEntry",
                    fieldbackground=COLORS["bg_input"],
                    foreground=COLORS["text_primary"],
                    insertcolor=COLORS["text_primary"],
                    bordercolor=COLORS["border_default"],
                    lightcolor=COLORS["bg_input"],
                    darkcolor=COLORS["bg_input"],
                    padding=SPACING["sm"])
    style.map("TEntry",
              bordercolor=[('focus', COLORS["border_focus"])],
              lightcolor=[('focus', COLORS["border_focus"])])
    
    # --- TCombobox ---
    style.configure("TCombobox",
                    fieldbackground=COLORS["bg_input"],
                    background=COLORS["bg_elevated"],
                    foreground=COLORS["text_primary"],
                    arrowcolor=COLORS["text_secondary"],
                    bordercolor=COLORS["border_default"],
                    lightcolor=COLORS["bg_input"],
                    darkcolor=COLORS["bg_input"],
                    padding=SPACING["sm"])
    style.map("TCombobox",
              fieldbackground=[('readonly', COLORS["bg_input"])],
              selectbackground=[('readonly', COLORS["primary"])],
              selectforeground=[('readonly', COLORS["text_inverse"])],
              bordercolor=[('focus', COLORS["border_focus"])])
    
    # --- TNotebook ---
    style.configure("TNotebook",
                    background=COLORS["bg_dark"],
                    bordercolor=COLORS["bg_dark"],
                    tabmargins=[SPACING["sm"], SPACING["sm"], SPACING["sm"], 0])
    
    style.configure("TNotebook.Tab",
                    background=COLORS["bg_surface"],
                    foreground=COLORS["text_secondary"],
                    padding=[SPACING["lg"], SPACING["sm"]],
                    font=(FONTS["family"], FONTS["size_base"]))
    style.map("TNotebook.Tab",
              background=[('selected', COLORS["bg_elevated"])],
              foreground=[('selected', COLORS["primary"])],
              expand=[('selected', [1, 1, 1, 0])])
    
    # --- TLabelframe ---
    style.configure("TLabelframe",
                    background=COLORS["bg_surface"],
                    bordercolor=COLORS["border_default"],
                    lightcolor=COLORS["border_default"],
                    darkcolor=COLORS["border_default"])
    style.configure("TLabelframe.Label",
                    background=COLORS["bg_surface"],
                    foreground=COLORS["primary"],
                    font=(FONTS["family"], FONTS["size_base"], "bold"))
    
    # --- TRadiobutton ---
    style.configure("TRadiobutton",
                    background=COLORS["bg_dark"],
                    foreground=COLORS["text_primary"],
                    indicatorbackground=COLORS["bg_input"],
                    indicatorforeground=COLORS["primary"],
                    font=(FONTS["family"], FONTS["size_base"]))
    style.map("TRadiobutton",
              background=[('active', COLORS["bg_dark"])],
              indicatorbackground=[('selected', COLORS["primary"])])
    
    style.configure("Card.TRadiobutton",
                    background=COLORS["bg_surface"])
    style.map("Card.TRadiobutton",
              background=[('active', COLORS["bg_surface"])])
    
    # --- TCheckbutton ---
    style.configure("TCheckbutton",
                    background=COLORS["bg_dark"],
                    foreground=COLORS["text_primary"],
                    indicatorbackground=COLORS["bg_input"],
                    indicatorforeground=COLORS["primary"],
                    font=(FONTS["family"], FONTS["size_base"]))
    
    # --- TProgressbar ---
    style.configure("TProgressbar",
                    background=COLORS["primary"],
                    troughcolor=COLORS["bg_input"],
                    bordercolor=COLORS["bg_input"],
                    lightcolor=COLORS["primary"],
                    darkcolor=COLORS["primary"])
    
    # --- Scrollbar ---
    style.configure("TScrollbar",
                    background=COLORS["bg_elevated"],
                    troughcolor=COLORS["bg_dark"],
                    bordercolor=COLORS["bg_dark"],
                    arrowcolor=COLORS["text_muted"])
    
    # --- Separator ---
    style.configure("TSeparator",
                    background=COLORS["border_default"])
    
    return style


def get_text_widget_config():
    """Return configuration dict for tk.Text widgets (not ttk)."""
    return {
        "bg": COLORS["bg_input"],
        "fg": COLORS["text_primary"],
        "insertbackground": COLORS["text_primary"],
        "selectbackground": COLORS["primary"],
        "selectforeground": COLORS["text_inverse"],
        "relief": "flat",
        "highlightthickness": 1,
        "highlightbackground": COLORS["border_default"],
        "highlightcolor": COLORS["border_focus"],
        "font": (FONTS["family"], FONTS["size_base"]),
        "padx": SPACING["sm"],
        "pady": SPACING["sm"],
    }


def get_listbox_config():
    """Return configuration dict for tk.Listbox widgets."""
    return {
        "bg": COLORS["bg_input"],
        "fg": COLORS["text_primary"],
        "selectbackground": COLORS["primary"],
        "selectforeground": COLORS["text_inverse"],
        "relief": "flat",
        "highlightthickness": 1,
        "highlightbackground": COLORS["border_default"],
        "highlightcolor": COLORS["border_focus"],
        "font": (FONTS["family"], FONTS["size_base"]),
    }
