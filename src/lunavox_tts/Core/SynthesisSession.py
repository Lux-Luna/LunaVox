"""
SynthesisSession - Encapsulates all state for a single TTS synthesis request.

This replaces the global `context` object to enable:
- Thread-safe concurrent synthesis
- Cleaner dependency injection
- Easier unit testing
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..Audio.ReferenceAudio import ReferenceAudio
    from ..ModelManager import GSVModel


@dataclass
class SynthesisSession:
    """
    Encapsulates all state required for a single TTS synthesis operation.
    
    This is passed through the inference pipeline instead of relying on
    global state, enabling concurrent synthesis and cleaner architecture.
    """
    
    # Required fields
    speaker: str
    language: str
    
    # Reference audio (set via set_reference_audio or load_persona)
    prompt_audio: Optional["ReferenceAudio"] = None
    
    # Model reference (resolved lazily by ModelManager)
    model: Optional["GSVModel"] = None
    
    # Model version (v2, v2Pro, v2ProPlus) - resolved from model metadata
    model_version: str = "v2"
    
    # Session flags
    is_persona_mode: bool = False
    skip_prompt_encoder: bool = False
    
    def validate(self) -> bool:
        """Validate that session has all required data for synthesis."""
        if not self.speaker:
            return False
        if self.prompt_audio is None:
            return False
        return True
    
    @classmethod
    def from_context(cls, speaker: str, language: str) -> "SynthesisSession":
        """
        Create a session from the legacy global context.
        
        This is a compatibility bridge during the migration period.
        """
        from ..Utils.Shared import context
        
        return cls(
            speaker=speaker or context.current_speaker,
            language=language or context.current_language,
            prompt_audio=context.current_prompt_audio,
        )


# Factory function for creating sessions
def create_session(
    speaker: str,
    language: str = "ja",
    prompt_audio: Optional["ReferenceAudio"] = None,
) -> SynthesisSession:
    """Create a new synthesis session with the given parameters."""
    return SynthesisSession(
        speaker=speaker,
        language=language,
        prompt_audio=prompt_audio,
    )
