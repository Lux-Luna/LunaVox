from rich.console import Console
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..Audio.ReferenceAudio import ReferenceAudio

console: Console = Console()


class Context:
    def __init__(self):
        self.current_speaker: str = ""
        self.current_prompt_audio: Optional["ReferenceAudio"] = None
        self.current_language: str = "ja"  # Supported: ja, en, zh


context: Context = Context()

