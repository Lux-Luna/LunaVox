"""
LunaVox Frontend Package.

Provides a unified interface for language-specific text processing.
"""
from typing import Dict, Type, Optional
from .BaseFrontend import AbstractFrontend

# Registry of available frontends
_frontend_registry: Dict[str, Type[AbstractFrontend]] = {}
_frontend_instances: Dict[str, AbstractFrontend] = {}


def register_frontend(language: str, frontend_class: Type[AbstractFrontend]) -> None:
    """
    Register a frontend class for a language.
    
    Args:
        language: Language code (e.g., 'en', 'zh', 'ja').
        frontend_class: The frontend class to register.
    """
    _frontend_registry[language.lower()] = frontend_class


def get_frontend(language: str) -> AbstractFrontend:
    """
    Get a frontend instance for the specified language.
    
    Uses lazy initialization and caching.
    
    Args:
        language: Language code (e.g., 'en', 'zh', 'ja').
        
    Returns:
        Frontend instance for the language.
        
    Raises:
        ValueError: If no frontend is registered for the language.
    """
    lang = language.lower()
    
    # Return cached instance if available
    if lang in _frontend_instances:
        return _frontend_instances[lang]
    
    # Try to get from registry
    if lang in _frontend_registry:
        instance = _frontend_registry[lang]()
        _frontend_instances[lang] = instance
        return instance
    
    # Auto-register built-in frontends on first access
    if lang == "en":
        from ...English.EnglishG2P import EnglishFrontend
        register_frontend("en", EnglishFrontend)
        return get_frontend("en")
    elif lang == "zh":
        from ...Chinese.ChineseG2P import ChineseFrontend
        register_frontend("zh", ChineseFrontend)
        return get_frontend("zh")
    elif lang == "ja":
        from ...Japanese.JapaneseG2P import JapaneseFrontend
        register_frontend("ja", JapaneseFrontend)
        return get_frontend("ja")
    
    raise ValueError(f"No frontend registered for language: {language}")


def list_available_languages() -> list:
    """Return list of supported language codes."""
    return ["en", "zh", "ja"]  # Built-in languages


__all__ = [
    "AbstractFrontend",
    "register_frontend",
    "get_frontend",
    "list_available_languages",
]
