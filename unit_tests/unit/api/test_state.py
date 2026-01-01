"""
Unit tests for lunavox_tts.API.state module.

Tests reference audio state management.
"""
import pytest


class TestReferenceAudioState:
    """Tests for reference audio configuration management."""
    
    def test_set_and_get_reference_audio(self):
        """Set configuration and retrieve it."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            get_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        config = {'audio_path': '/test/path.wav', 'audio_text': 'Hello'}
        set_reference_audio_config("test_char", config)
        
        retrieved = get_reference_audio("test_char")
        assert retrieved == config
    
    def test_get_nonexistent_returns_none(self):
        """Getting non-existent character returns None."""
        from lunavox_tts.API.state import get_reference_audio, clear_all_reference_audio
        
        clear_all_reference_audio()
        
        result = get_reference_audio("nonexistent")
        assert result is None
    
    def test_has_reference_audio(self):
        """has_reference_audio works correctly."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        assert has_reference_audio("test") is False
        
        set_reference_audio_config("test", {'audio_path': 'test.wav'})
        assert has_reference_audio("test") is True
    
    def test_clear_all(self):
        """clear_all_reference_audio removes all configurations."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio,
            clear_all_reference_audio
        )
        
        set_reference_audio_config("char1", {'audio_path': 'test1.wav'})
        set_reference_audio_config("char2", {'audio_path': 'test2.wav'})
        
        clear_all_reference_audio()
        
        assert has_reference_audio("char1") is False
        assert has_reference_audio("char2") is False
    
    def test_overwrite_existing(self):
        """Setting same character overwrites existing config."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            get_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        set_reference_audio_config("char", {'audio_path': 'old.wav'})
        set_reference_audio_config("char", {'audio_path': 'new.wav'})
        
        config = get_reference_audio("char")
        assert config['audio_path'] == 'new.wav'


class TestNormalizeLanguage:
    """Tests for normalize_language function."""
    
    def test_supported_languages(self):
        """Supported language codes are passed through."""
        from lunavox_tts.API.state import normalize_language
        
        assert normalize_language("ja") == "ja"
        assert normalize_language("en") == "en"
        assert normalize_language("zh") == "zh"
    
    def test_case_insensitive(self):
        """Language normalization is case-insensitive."""
        from lunavox_tts.API.state import normalize_language
        
        assert normalize_language("JA") == "ja"
        assert normalize_language("EN") == "en"
        assert normalize_language("ZH") == "zh"
    
    def test_unsupported_defaults_to_ja(self):
        """Unsupported languages default to 'ja'."""
        from lunavox_tts.API.state import normalize_language
        
        assert normalize_language("fr") == "ja"
        assert normalize_language("de") == "ja"
        assert normalize_language("unknown") == "ja"
    
    def test_none_defaults_to_ja(self):
        """None input defaults to 'ja'."""
        from lunavox_tts.API.state import normalize_language
        
        assert normalize_language(None) == "ja"


class TestSupportedAudioExts:
    """Tests for SUPPORTED_AUDIO_EXTS constant."""
    
    def test_wav_supported(self):
        """Common audio formats are supported."""
        from lunavox_tts.API.state import SUPPORTED_AUDIO_EXTS
        
        assert '.wav' in SUPPORTED_AUDIO_EXTS
        assert '.flac' in SUPPORTED_AUDIO_EXTS
        assert '.ogg' in SUPPORTED_AUDIO_EXTS
    
    def test_case_sensitive(self):
        """Extensions are stored lowercase."""
        from lunavox_tts.API.state import SUPPORTED_AUDIO_EXTS
        
        # Verify all extensions are lowercase
        for ext in SUPPORTED_AUDIO_EXTS:
            assert ext == ext.lower()
