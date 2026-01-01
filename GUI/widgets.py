"""
LunaVox GUI Widgets - Reusable UI components
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from theme import COLORS, FONTS, SPACING


class AnimatedButton(ttk.Button):
    """Button with hover animation effects."""
    
    def __init__(self, parent, **kwargs):
        self.command_callback = kwargs.pop('command', None)
        super().__init__(parent, command=self._on_click, **kwargs)
        
        self._is_loading = False
        self._original_text = kwargs.get('text', '')
        
        # Bind hover events
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
    def _on_enter(self, event):
        """Handle mouse enter."""
        if not self._is_loading:
            self.configure(cursor='hand2')
            
    def _on_leave(self, event):
        """Handle mouse leave."""
        self.configure(cursor='')
        
    def _on_click(self):
        """Handle click with optional callback."""
        if self.command_callback and not self._is_loading:
            self.command_callback()
            
    def set_loading(self, loading: bool, text: str = ""):
        """Set loading state with optional text."""
        self._is_loading = loading
        if loading:
            self.configure(state='disabled', text=text or "...")
        else:
            self.configure(state='normal', text=self._original_text)
            
    def set_text(self, text: str):
        """Update button text."""
        self._original_text = text
        if not self._is_loading:
            self.configure(text=text)


class StatusIndicator(ttk.Frame):
    """Visual status indicator with icon and text."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")
        
        self.icon_label = ttk.Label(self, text="", style="TLabel")
        self.icon_label.pack(side="left")
        
        self.text_label = ttk.Label(self, text="", style="Muted.TLabel")
        self.text_label.pack(side="left", padx=(4, 0))
        
    def set_status(self, status: str, icon: str = "", color: str = None):
        """Update status display."""
        self.icon_label.configure(text=icon)
        self.text_label.configure(text=status)
        if color:
            self.text_label.configure(foreground=color)
            self.icon_label.configure(foreground=color)
            
    def set_loading(self):
        """Set to loading state."""
        self.set_status("Loading...", "⌛", COLORS["warning"])
        
    def set_success(self, text: str = "Success"):
        """Set to success state."""
        self.set_status(text, "✓", COLORS["success"])
        
    def set_error(self, text: str = "Error"):
        """Set to error state."""
        self.set_status(text, "✗", COLORS["error"])
        
    def clear(self):
        """Clear status display."""
        self.set_status("", "")


class LatencyDisplay(ttk.Frame):
    """Display for inference latency metrics."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="Card.TFrame")
        
        # Create a compact display
        self.container = ttk.Frame(self, style="Card.TFrame")
        self.container.pack(fill="x", padx=SPACING["sm"], pady=SPACING["xs"])
        
        ttk.Label(
            self.container, 
            text="⏱", 
            style="Card.TLabel",
            font=(FONTS["family"], FONTS["size_lg"])
        ).pack(side="left")
        
        self.latency_var = tk.StringVar(value="--")
        self.latency_label = ttk.Label(
            self.container,
            textvariable=self.latency_var,
            style="Card.TLabel",
            font=(FONTS["family"], FONTS["size_base"], "bold")
        )
        self.latency_label.pack(side="left", padx=(4, 0))
        
        ttk.Label(
            self.container,
            text="ms",
            style="Muted.TLabel"
        ).pack(side="left", padx=(2, SPACING["md"]))
        
        # RTF (Real-time Factor)
        ttk.Label(
            self.container,
            text="RTF:",
            style="Muted.TLabel"
        ).pack(side="left")
        
        self.rtf_var = tk.StringVar(value="--")
        self.rtf_label = ttk.Label(
            self.container,
            textvariable=self.rtf_var,
            style="Card.TLabel",
            font=(FONTS["family"], FONTS["size_base"], "bold")
        )
        self.rtf_label.pack(side="left", padx=(4, 0))
        
    def update(self, latency_ms: float, audio_duration_ms: float = None):
        """Update latency display."""
        self.latency_var.set(f"{latency_ms:.0f}")
        
        if audio_duration_ms and audio_duration_ms > 0:
            rtf = latency_ms / audio_duration_ms
            self.rtf_var.set(f"{rtf:.2f}x")
            
            # Color code RTF
            if rtf < 1.0:
                self.rtf_label.configure(foreground=COLORS["success"])
            elif rtf < 2.0:
                self.rtf_label.configure(foreground=COLORS["warning"])
            else:
                self.rtf_label.configure(foreground=COLORS["error"])
        else:
            self.rtf_var.set("--")
            
    def clear(self):
        """Clear display."""
        self.latency_var.set("--")
        self.rtf_var.set("--")
        self.rtf_label.configure(foreground=COLORS["text_primary"])


class AudioOutputPanel(ttk.Frame):
    """Panel for displaying and saving audio output."""
    
    def __init__(self, parent, on_save: Callable = None, on_play: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="Card.TFrame")
        
        self.on_save = on_save
        self.on_play = on_play
        self.current_audio_path = None
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the audio output panel UI."""
        # Header
        header = ttk.Frame(self, style="Card.TFrame")
        header.pack(fill="x", padx=SPACING["sm"], pady=SPACING["sm"])
        
        ttk.Label(
            header,
            text="🔊 Output",
            style="Card.TLabel",
            font=(FONTS["family"], FONTS["size_base"], "bold")
        ).pack(side="left")
        
        # Buttons
        btn_frame = ttk.Frame(self, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=SPACING["sm"], pady=(0, SPACING["sm"]))
        
        self.btn_play = ttk.Button(
            btn_frame,
            text="▶ Play",
            command=self._on_play_click
        )
        self.btn_play.pack(side="left", padx=(0, SPACING["xs"]))
        self.btn_play.configure(state="disabled")
        
        self.btn_save = ttk.Button(
            btn_frame,
            text="💾 Save As...",
            command=self._on_save_click
        )
        self.btn_save.pack(side="left", padx=(0, SPACING["xs"]))
        self.btn_save.configure(state="disabled")
        
        self.btn_open_folder = ttk.Button(
            btn_frame,
            text="📂 Open Folder",
            command=self._on_open_folder_click
        )
        self.btn_open_folder.pack(side="left")
        self.btn_open_folder.configure(state="disabled")
        
        # File info
        self.file_info_var = tk.StringVar(value="No audio generated yet")
        self.file_info = ttk.Label(
            self,
            textvariable=self.file_info_var,
            style="Muted.TLabel"
        )
        self.file_info.pack(anchor="w", padx=SPACING["sm"], pady=(0, SPACING["sm"]))
        
    def set_audio(self, audio_path: str, duration_sec: float = None):
        """Set the current audio file."""
        import os
        self.current_audio_path = audio_path
        
        if audio_path and os.path.exists(audio_path):
            filename = os.path.basename(audio_path)
            size_kb = os.path.getsize(audio_path) / 1024
            
            info = f"{filename} ({size_kb:.1f} KB"
            if duration_sec:
                info += f", {duration_sec:.1f}s"
            info += ")"
            
            self.file_info_var.set(info)
            self.btn_play.configure(state="normal")
            self.btn_save.configure(state="normal")
            self.btn_open_folder.configure(state="normal")
        else:
            self.file_info_var.set("No audio generated yet")
            self.btn_play.configure(state="disabled")
            self.btn_save.configure(state="disabled")
            self.btn_open_folder.configure(state="disabled")
            
    def _on_play_click(self):
        """Handle play button click."""
        if self.on_play and self.current_audio_path:
            self.on_play(self.current_audio_path)
            
    def _on_save_click(self):
        """Handle save button click."""
        if self.on_save and self.current_audio_path:
            self.on_save(self.current_audio_path)
            
    def _on_open_folder_click(self):
        """Open the folder containing the audio file."""
        import os
        import subprocess
        import sys
        
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            folder = os.path.dirname(self.current_audio_path)
            if sys.platform == 'darwin':
                subprocess.run(['open', folder])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', folder])
            else:
                subprocess.run(['xdg-open', folder])


class HoverFrame(ttk.Frame):
    """Frame with hover effect."""
    
    def __init__(self, parent, hover_style: str = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.default_style = kwargs.get('style', 'TFrame')
        self.hover_style = hover_style or self.default_style
        
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
    def _on_enter(self, event):
        self.configure(style=self.hover_style)
        
    def _on_leave(self, event):
        self.configure(style=self.default_style)


class ProgressPanel(ttk.Frame):
    """Panel showing progress with percentage and ETA."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="Card.TFrame")
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self,
            variable=self.progress_var,
            mode="determinate",
            length=300
        )
        self.progress_bar.pack(fill="x", padx=SPACING["sm"], pady=SPACING["sm"])
        
        # Status text
        info_frame = ttk.Frame(self, style="Card.TFrame")
        info_frame.pack(fill="x", padx=SPACING["sm"], pady=(0, SPACING["sm"]))
        
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(
            info_frame,
            textvariable=self.status_var,
            style="Card.TLabel"
        )
        self.status_label.pack(side="left")
        
        self.percent_var = tk.StringVar(value="")
        self.percent_label = ttk.Label(
            info_frame,
            textvariable=self.percent_var,
            style="Muted.TLabel"
        )
        self.percent_label.pack(side="right")
        
    def set_progress(self, value: float, status: str = ""):
        """Update progress (0-100)."""
        self.progress_var.set(value)
        self.percent_var.set(f"{value:.0f}%")
        if status:
            self.status_var.set(status)
            
    def set_indeterminate(self, status: str = ""):
        """Set to indeterminate mode."""
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(10)
        self.percent_var.set("")
        if status:
            self.status_var.set(status)
            
    def stop(self):
        """Stop and reset progress."""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_var.set(0)
        self.percent_var.set("")
        self.status_var.set("")
