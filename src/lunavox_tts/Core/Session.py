"""
Synthesis Session - Encapsulates all state for a single TTS request.

Replaces global `context` for thread-safe concurrent synthesis.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..Resources.Audio.ReferenceAudio import ReferenceAudio
    from ..ModelManager import GSVModel


@dataclass
class SynthesisSession:
    """Encapsulates all state for a single TTS synthesis operation."""
    
    speaker: str
    language: str
    prompt_audio: Optional["ReferenceAudio"] = None
    model: Optional["GSVModel"] = None
    model_version: str = "v2"
    is_persona_mode: bool = False
    skip_prompt_encoder: bool = False
    
    def validate(self) -> bool:
        if not self.speaker:
            return False
        if self.prompt_audio is None:
            return False
        return True


def create_session(
    speaker: str,
    language: str = "ja",
    prompt_audio: Optional["ReferenceAudio"] = None,
) -> SynthesisSession:
    """Create a new synthesis session."""
    return SynthesisSession(speaker=speaker, language=language, prompt_audio=prompt_audio)

