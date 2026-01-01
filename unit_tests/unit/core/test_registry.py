"""
Unit tests for lunavox_tts.Core.Frontend.registry module.

Tests LanguageRegistry and language frontend management.
"""
import pytest
from unittest.mock import MagicMock


class TestLanguageRegistry:
    """Tests for LanguageRegistry class."""
    
    def test_register_and_get(self):
        """Registered frontend can be retrieved."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        mock_frontend_class = MagicMock(return_value=MagicMock())
        
        registry.register("test", mock_frontend_class)
        frontend = registry.get("test")
        
        assert frontend is not None
        mock_frontend_class.assert_called_once()
    
    def test_unsupported_language_raises(self):
        """Getting unsupported language raises ValueError."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        
        with pytest.raises(ValueError, match="Unsupported language"):
            registry.get("unsupported_lang_xyz")
    
    def test_case_insensitive(self):
        """Language codes are case-insensitive."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        mock_frontend_class = MagicMock(return_value=MagicMock())
        
        registry.register("TEST", mock_frontend_class)
        
        # Should be retrievable with lowercase
        frontend1 = registry.get("test")
        frontend2 = registry.get("TEST")
        frontend3 = registry.get("Test")
        
        # All should return the same instance
        assert frontend1 is frontend2
        assert frontend2 is frontend3
    
    def test_singleton_instance(self):
        """Multiple get calls return same instance."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        mock_instance = MagicMock()
        mock_frontend_class = MagicMock(return_value=mock_instance)
        
        registry.register("singleton", mock_frontend_class)
        
        frontend1 = registry.get("singleton")
        frontend2 = registry.get("singleton")
        
        # Class should only be instantiated once
        mock_frontend_class.assert_called_once()
        assert frontend1 is frontend2
    
    def test_list_languages_includes_registered(self):
        """list_languages includes registered languages."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        mock_frontend_class = MagicMock(return_value=MagicMock())
        
        registry.register("custom", mock_frontend_class)
        
        languages = registry.list_languages()
        assert "custom" in languages
    
    def test_is_supported(self):
        """is_supported returns True for registered languages."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        mock_frontend_class = MagicMock(return_value=MagicMock())
        
        registry.register("supported", mock_frontend_class)
        
        assert registry.is_supported("supported") is True
        assert registry.is_supported("not_registered") is False
    
    def test_re_register_clears_instance(self):
        """Re-registering a language clears the cached instance."""
        from lunavox_tts.Core.Frontend.registry import LanguageRegistry
        
        registry = LanguageRegistry()
        
        mock_instance_1 = MagicMock()
        mock_instance_2 = MagicMock()
        mock_class_1 = MagicMock(return_value=mock_instance_1)
        mock_class_2 = MagicMock(return_value=mock_instance_2)
        
        registry.register("lang", mock_class_1)
        frontend1 = registry.get("lang")
        
        # Re-register with new class
        registry.register("lang", mock_class_2)
        frontend2 = registry.get("lang")
        
        # Should now get the new instance
        assert frontend1 is not frontend2
        assert frontend2 is mock_instance_2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_language_frontend(self):
        """get_language_frontend uses global registry."""
        from lunavox_tts.Core.Frontend.registry import (
            get_language_frontend, 
            register_language,
            language_registry
        )
        
        mock_frontend = MagicMock()
        mock_class = MagicMock(return_value=mock_frontend)
        
        register_language("convenience_test", mock_class)
        frontend = get_language_frontend("convenience_test")
        
        assert frontend is mock_frontend
    
    def test_list_supported_languages(self):
        """list_supported_languages works correctly."""
        from lunavox_tts.Core.Frontend.registry import (
            list_supported_languages,
            register_language
        )
        
        mock_class = MagicMock(return_value=MagicMock())
        register_language("list_test", mock_class)
        
        languages = list_supported_languages()
        assert "list_test" in languages
