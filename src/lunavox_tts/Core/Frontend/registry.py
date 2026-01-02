"""
Language Registry - Centralized registry for language frontends.

Enables dynamic language plugin registration at runtime.
"""
import logging
from typing import TYPE_CHECKING, Dict, Type, Optional, List

if TYPE_CHECKING:
    from .base import AbstractFrontend

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Central registry for language-specific frontends.
    
    Uses lazy import to avoid loading heavy dependencies (jieba, pyopenjtalk)
    until a language is actually requested.
    """
    
    # Builtin frontend paths (module:class format)
    _BUILTIN_FRONTENDS = {
        'en': 'lunavox_tts.Languages.English.EnglishG2P:EnglishFrontend',
        'zh': 'lunavox_tts.Languages.Chinese.ChineseG2P:ChineseFrontend',
        'ja': 'lunavox_tts.Languages.Japanese.JapaneseG2P:JapaneseFrontend',
    }
    
    def __init__(self):
        # Store module paths as strings for lazy import
        self._frontend_paths: Dict[str, str] = dict(self._BUILTIN_FRONTENDS)
        self._instances: Dict[str, "AbstractFrontend"] = {}
    
    def register(self, language: str, frontend_class: Type["AbstractFrontend"]) -> None:
        """Register a frontend class directly (for plugins)."""
        lang = language.lower()
        # Store the class directly, not as path
        self._frontend_paths[lang] = frontend_class
        if lang in self._instances:
            del self._instances[lang]
        logger.debug(f"Registered frontend for language: {lang}")
    
    def get(self, language: str) -> "AbstractFrontend":
        """Get frontend instance, lazily importing if needed."""
        lang = language.lower()
        
        if lang in self._instances:
            return self._instances[lang]
        
        if lang not in self._frontend_paths:
            raise ValueError(f"Unsupported language: {language}. Available: {self.list_languages()}")
        
        frontend_ref = self._frontend_paths[lang]
        
        # Handle both string paths and direct class references
        if isinstance(frontend_ref, str):
            frontend_class = self._import_frontend(frontend_ref)
        else:
            frontend_class = frontend_ref
        
        instance = frontend_class()
        self._instances[lang] = instance
        return instance
    
    def _import_frontend(self, path: str) -> Type["AbstractFrontend"]:
        """Dynamically import a frontend class from module:class path."""
        import importlib
        module_path, class_name = path.rsplit(':', 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as e:
            logger.error(f"Failed to import frontend from {path}: {e}")
            raise
    
    def list_languages(self) -> List[str]:
        return list(self._frontend_paths.keys())
    
    def is_supported(self, language: str) -> bool:
        return language.lower() in self._frontend_paths
    
    def clear_instances(self) -> None:
        """Clear cached frontend instances to release memory."""
        self._instances.clear()
        logger.debug("Frontend instances cleared.")


# Global registry instance
language_registry = LanguageRegistry()


# Convenience functions
def get_language_frontend(language: str) -> "AbstractFrontend":
    return language_registry.get(language)


def register_language(language: str, frontend_class: Type["AbstractFrontend"]) -> None:
    language_registry.register(language, frontend_class)


def list_supported_languages() -> List[str]:
    return language_registry.list_languages()
