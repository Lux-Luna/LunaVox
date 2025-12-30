from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from ..Audio.ReferenceAudio import ReferenceAudio

# Replace rich.console with standard logging
logger = logging.getLogger("LunaVox")

class ConsoleShim:
    """Shim for rich.console.Console to use standard logging."""
    def print(self, *args, **kwargs):
        # Join args with space if multiple arguments, similar to print
        msg = " ".join(str(arg) for arg in args)
        logger.info(msg)

console = ConsoleShim()


class Context:
    def __init__(self):
        self.current_speaker: str = ""
        self.current_prompt_audio: Optional["ReferenceAudio"] = None
        self.current_language: str = "ja"  # Supported: ja, en, zh


context: Context = Context()
