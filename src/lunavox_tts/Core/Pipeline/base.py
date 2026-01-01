"""
InferencePipeline - Base class for model-specific inference pipelines.

This abstraction allows different model versions (v2, v2Pro, v2ProPlus)
to have their own optimized inference paths while sharing common interfaces.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from ..SynthesisSession import SynthesisSession
    from ...ModelManager import GSVModel


class InferencePipeline(ABC):
    """
    Abstract base class for TTS inference pipelines.
    
    Each model version can implement its own optimized pipeline
    while conforming to this common interface.
    """
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version this pipeline handles."""
        pass
    
    @abstractmethod
    def run(
        self,
        text: str,
        session: "SynthesisSession",
        model: "GSVModel",
    ) -> Optional[np.ndarray]:
        """
        Execute the full TTS pipeline.
        
        Args:
            text: Input text to synthesize.
            session: Synthesis session containing state.
            model: Loaded model sessions.
            
        Returns:
            Audio waveform as numpy array, or None on failure.
        """
        pass
    
    def preprocess_text(self, text: str, language: str) -> str:
        """
        Apply text preprocessing (punctuation padding, etc.).
        
        This default implementation adds leading/trailing punctuation
        to prevent sentence boundary issues.
        """
        from ..Processors.text_processor import preprocess_text
        return preprocess_text(text, language)
    
    def postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio postprocessing (NaN removal, silence padding, etc.).
        """
        from ..Processors.audio_processor import postprocess_audio
        return postprocess_audio(audio)
