"""
Language Registry - Centralized registry for language frontends.

This enables dynamic language plugin registration at runtime without
modifying core inference code.
"""
import logging
from typing import TYPE_CHECKING, Dict, Type, Optional, List

if TYPE_CHECKING:
    from .Frontend.BaseFrontend import AbstractFrontend

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """
    Central registry for language-specific frontends.
    
    Supports:
    - Runtime registration of new language frontends
    - Lazy loading of built-in frontends
    - Querying available languages
    """
    
    def __init__(self):
        self._frontends: Dict[str, Type["AbstractFrontend"]] = {}
        self._instances: Dict[str, "AbstractFrontend"] = {}
        self._builtin_registered = False
    
    def register(self, language: str, frontend_class: Type["AbstractFrontend"]) -> None:
        """
        Register a frontend class for a language.
        
        Args:
            language: Language code (e.g., 'en', 'zh', 'ja').
            frontend_class: The frontend class to register.
        """
        lang = language.lower()
        self._frontends[lang] = frontend_class
        # Clear cached instance if re-registering
        if lang in self._instances:
            del self._instances[lang]
        logger.debug(f"Registered frontend for language: {lang}")
    
    def get(self, language: str) -> "AbstractFrontend":
        """
        Get a frontend instance for the specified language.
        
        Uses lazy initialization and caching.
        
        Args:
            language: Language code.
            
        Returns:
            Frontend instance.
            
        Raises:
            ValueError: If language is not supported.
        """
        lang = language.lower()
        
        # Return cached instance
        if lang in self._instances:
            return self._instances[lang]
        
        # Ensure built-in frontends are registered
        if not self._builtin_registered:
            self._register_builtins()
        
        # Create instance
        if lang in self._frontends:
            instance = self._frontends[lang]()
            self._instances[lang] = instance
            return instance
        
        raise ValueError(f"Unsupported language: {language}. Available: {self.list_languages()}")
    
    def _register_builtins(self) -> None:
        """Register built-in language frontends."""
        if self._builtin_registered:
            return
        
        try:
            from ..English.EnglishG2P import EnglishFrontend
            self.register("en", EnglishFrontend)
        except ImportError:
            logger.debug("English frontend not available")
        
        try:
            from ..Chinese.ChineseG2P import ChineseFrontend
            self.register("zh", ChineseFrontend)
        except ImportError:
            logger.debug("Chinese frontend not available")
        
        try:
            from ..Japanese.JapaneseG2P import JapaneseFrontend
            self.register("ja", JapaneseFrontend)
        except ImportError:
            logger.debug("Japanese frontend not available")
        
        self._builtin_registered = True
    
    def list_languages(self) -> List[str]:
        """List all registered language codes."""
        if not self._builtin_registered:
            self._register_builtins()
        return list(self._frontends.keys())
    
    def is_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        if not self._builtin_registered:
            self._register_builtins()
        return language.lower() in self._frontends


# Global registry instance
language_registry = LanguageRegistry()


# Convenience functions
def get_language_frontend(language: str) -> "AbstractFrontend":
    """Get a frontend for the specified language."""
    return language_registry.get(language)


def register_language(language: str, frontend_class: Type["AbstractFrontend"]) -> None:
    """Register a new language frontend."""
    language_registry.register(language, frontend_class)


def list_supported_languages() -> List[str]:
    """List all supported languages."""
    return language_registry.list_languages()
