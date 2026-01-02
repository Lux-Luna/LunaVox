"""
LoadingContext - Intent-driven model loading configuration.

Replaces the "guess-and-heal" pattern with explicit configuration.
Users specify their intended mode upfront, ensuring optimal resource loading.
"""
from dataclasses import dataclass
from typing import Literal, Optional, Set


@dataclass
class LoadingContext:
    """
    Configuration object for intent-driven model loading.
    
    Instead of loading everything and unloading unused components,
    this context specifies exactly what should be loaded.
    
    Modes:
        - reference: Full model with all components (for reference audio TTS)
        - persona_with_ge: Persona mode but global_emb not cached (needs PE)
        - persona_no_ge: Persona mode with cached global_emb (skips PE)
    
    Usage:
        context = LoadingContext(
            character_name="luna",
            model_dir="/path/to/model",
            mode="persona_no_ge"
        )
        model_manager.load_character_with_context(context)
    """
    character_name: str
    model_dir: str
    mode: Literal["reference", "persona_with_ge", "persona_no_ge"]
    version: Optional[str] = None  # Auto-detected if None
    
    def get_skip_components(self) -> Set[str]:
        """
        Return the set of components to skip during loading.
        
        Returns:
            Set of component names to skip.
        """
        if self.mode == "persona_no_ge":
            return {"PROMPT_ENCODER"}
        return set()
    
    @property
    def skip_prompt_encoder(self) -> bool:
        """Convenience property for backward compatibility."""
        return self.mode == "persona_no_ge"
    
    @classmethod
    def for_persona(cls, character_name: str, model_dir: str, has_global_emb: bool) -> "LoadingContext":
        """
        Factory method for Persona mode.
        
        Args:
            character_name: Name of the character.
            model_dir: Path to model directory.
            has_global_emb: Whether the persona has cached global embeddings.
            
        Returns:
            LoadingContext configured for optimal persona loading.
        """
        mode = "persona_no_ge" if has_global_emb else "persona_with_ge"
        return cls(character_name=character_name, model_dir=model_dir, mode=mode)
    
    @classmethod
    def for_reference(cls, character_name: str, model_dir: str) -> "LoadingContext":
        """
        Factory method for Reference audio mode.
        
        Args:
            character_name: Name of the character.
            model_dir: Path to model directory.
            
        Returns:
            LoadingContext configured for reference audio TTS.
        """
        return cls(character_name=character_name, model_dir=model_dir, mode="reference")
