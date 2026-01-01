from typing import TYPE_CHECKING, Optional
import logging
import warnings

if TYPE_CHECKING:
    from ..Resources.Audio.ReferenceAudio import ReferenceAudio

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
    """
    DEPRECATED: Use SynthesisSession instead.
    
    This global context is maintained for backward compatibility only.
    New code should use SynthesisSession for state management.
    """
    def __init__(self):
        self.current_speaker: str = ""
        self.current_prompt_audio: Optional["ReferenceAudio"] = None
        self.current_language: str = "ja"  # Supported: ja, en, zh
    
    def __setattr__(self, name, value):
        # Warn on first usage after deprecation
        if not hasattr(self, '_warned'):
            object.__setattr__(self, '_warned', set())
        if name not in getattr(self, '_warned', set()) and name != '_warned':
            self._warned.add(name)
            # Deprecation warning enabled - migrate to SynthesisSession
            warnings.warn(
                f"Global context.{name} is deprecated. Use SynthesisSession instead.",
                DeprecationWarning,
                stacklevel=3
            )
        object.__setattr__(self, name, value)


context: Context = Context()

