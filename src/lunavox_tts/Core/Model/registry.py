"""
Model Registry - Manages model metadata and paths.

First layer of the three-tier architecture:
- Registry: Tracks metadata (this module)
- Loader: Handles ONNX loading
- ModelManager: High-level facade
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .spec import detect_model_version, get_model_spec, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Metadata entry for a loaded/registered model."""
    name: str
    path: str
    version: str
    spec: ModelSpec
    is_loaded: bool = False
    components_loaded: Set[str] = field(default_factory=set)
    # State Healing: hint from API layer that prompt_encoder is not needed
    skip_prompt_encoder_hint: bool = False


class ModelRegistry:
    """Registry for tracking model metadata and paths."""
    
    def __init__(self):
        self._entries: Dict[str, ModelEntry] = {}
    
    def register(self, name: str, path: str, force_version: Optional[str] = None) -> ModelEntry:
        name = name.lower()
        
        old_hint = False
        if name in self._entries:
            entry = self._entries[name]
            old_hint = entry.skip_prompt_encoder_hint
            if entry.path == path and entry.path != "":
                return entry
            logger.debug(f"Re-registering model '{name}' with new path")
        
        version = force_version or detect_model_version(path)
        spec = get_model_spec(version)
        
        entry = ModelEntry(name=name, path=path, version=version, spec=spec, skip_prompt_encoder_hint=old_hint)
        self._entries[name] = entry
        
        logger.debug(f"Registered model '{name}': version={version}, path={path}")
        return entry
    
    def get(self, name: str) -> Optional[ModelEntry]:
        return self._entries.get(name.lower())
    
    def has(self, name: str) -> bool:
        return name.lower() in self._entries
    
    def unregister(self, name: str) -> bool:
        name = name.lower()
        if name in self._entries:
            del self._entries[name]
            return True
        return False
    
    def list_all(self) -> list:
        return list(self._entries.keys())
    
    def mark_loaded(self, name: str, components: Optional[Set[str]] = None) -> None:
        name = name.lower()
        if name in self._entries:
            self._entries[name].is_loaded = True
            if components:
                self._entries[name].components_loaded = components
    
    def set_optimization_hint(self, name: str, skip_prompt_encoder: bool) -> None:
        """Set optimization hint for a character (called by API.load_persona)."""
        name = name.lower()
        if name not in self._entries:
            # Pre-register a placeholder entry for the hint
            self._entries[name] = ModelEntry(
                name=name, path="", version="", spec=None,
                skip_prompt_encoder_hint=skip_prompt_encoder
            )
        else:
            self._entries[name].skip_prompt_encoder_hint = skip_prompt_encoder
        logger.debug(f"Set optimization hint for '{name}': skip_prompt_encoder={skip_prompt_encoder}")
    
    def get_optimization_hint(self, name: str) -> bool:
        """Get optimization hint for a character."""
        entry = self._entries.get(name.lower())
        return entry.skip_prompt_encoder_hint if entry else False


# Global registry instance
model_registry = ModelRegistry()
