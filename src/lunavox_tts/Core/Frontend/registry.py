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
    """Central registry for language-specific frontends."""
    
    def __init__(self):
        self._frontends: Dict[str, Type["AbstractFrontend"]] = {}
        self._instances: Dict[str, "AbstractFrontend"] = {}
        self._builtin_registered = False
    
    def register(self, language: str, frontend_class: Type["AbstractFrontend"]) -> None:
        lang = language.lower()
        self._frontends[lang] = frontend_class
        if lang in self._instances:
            del self._instances[lang]
        logger.debug(f"Registered frontend for language: {lang}")
    
    def get(self, language: str) -> "AbstractFrontend":
        lang = language.lower()
        
        if lang in self._instances:
            return self._instances[lang]
        
        if not self._builtin_registered:
            self._register_builtins()
        
        if lang in self._frontends:
            instance = self._frontends[lang]()
            self._instances[lang] = instance
            return instance
        
        raise ValueError(f"Unsupported language: {language}. Available: {self.list_languages()}")
    
    def _register_builtins(self) -> None:
        if self._builtin_registered:
            return
        
        try:
            from ...English.EnglishG2P import EnglishFrontend
            self.register("en", EnglishFrontend)
        except ImportError:
            logger.debug("English frontend not available")
        
        try:
            from ...Chinese.ChineseG2P import ChineseFrontend
            self.register("zh", ChineseFrontend)
        except ImportError:
            logger.debug("Chinese frontend not available")
        
        try:
            from ...Japanese.JapaneseG2P import JapaneseFrontend
            self.register("ja", JapaneseFrontend)
        except ImportError:
            logger.debug("Japanese frontend not available")
        
        self._builtin_registered = True
    
    def list_languages(self) -> List[str]:
        if not self._builtin_registered:
            self._register_builtins()
        return list(self._frontends.keys())
    
    def is_supported(self, language: str) -> bool:
        if not self._builtin_registered:
            self._register_builtins()
        return language.lower() in self._frontends


# Global registry instance
language_registry = LanguageRegistry()


# Convenience functions
def get_language_frontend(language: str) -> "AbstractFrontend":
    return language_registry.get(language)


def register_language(language: str, frontend_class: Type["AbstractFrontend"]) -> None:
    language_registry.register(language, frontend_class)


def list_supported_languages() -> List[str]:
    return language_registry.list_languages()
