"""
Unit tests for memory lifecycle management.

Tests cleanup cascade logic between unload_character, RuntimeManager,
and API state to prevent memory leaks.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRemoveReferenceAudio:
    """Tests for the new remove_reference_audio function."""
    
    def test_remove_existing_character(self):
        """Removing an existing character returns True."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            remove_reference_audio,
            has_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        set_reference_audio_config("char_to_remove", {'audio_path': 'test.wav'})
        assert has_reference_audio("char_to_remove") is True
        
        result = remove_reference_audio("char_to_remove")
        
        assert result is True
        assert has_reference_audio("char_to_remove") is False
    
    def test_remove_nonexistent_character(self):
        """Removing a nonexistent character returns False."""
        from lunavox_tts.API.state import (
            remove_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        result = remove_reference_audio("does_not_exist")
        assert result is False


class TestUnloadCharacterCleansState:
    """Tests verifying unload_character clears API state."""
    
    @patch("lunavox_tts.API.characters.model_manager")
    def test_unload_clears_reference_audio(self, mock_mm):
        """unload_character removes associated reference audio config."""
        from lunavox_tts.API.characters import unload_character
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        # Setup: character has reference audio
        set_reference_audio_config("test_char", {'audio_path': 'test.wav'})
        assert has_reference_audio("test_char") is True
        
        # Action
        unload_character("test_char")
        
        # Verify API state is cleared
        assert has_reference_audio("test_char") is False
        mock_mm.remove_character.assert_called_once_with(character_name="test_char")
    
    @patch("lunavox_tts.API.characters.model_manager")
    def test_unload_other_characters_preserved(self, mock_mm):
        """Unloading one character doesn't affect others."""
        from lunavox_tts.API.characters import unload_character
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio,
            clear_all_reference_audio
        )
        
        clear_all_reference_audio()
        
        # Setup: two characters
        set_reference_audio_config("char_a", {'audio_path': 'a.wav'})
        set_reference_audio_config("char_b", {'audio_path': 'b.wav'})
        
        # Action: unload only char_a
        unload_character("char_a")
        
        # Verify: char_a gone, char_b still present
        assert has_reference_audio("char_a") is False
        assert has_reference_audio("char_b") is True


class TestCleanupAllClearsApiState:
    """Tests for RuntimeManager.cleanup_all API state clearing."""
    
    def test_cleanup_all_clears_reference_audios(self):
        """cleanup_all() clears all reference audio configurations."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio
        )
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        # Setup: add multiple characters
        set_reference_audio_config("char1", {'audio_path': 'test1.wav'})
        set_reference_audio_config("char2", {'audio_path': 'test2.wav'})
        
        # Action
        runtime_manager.cleanup_all()
        
        # Verify
        assert has_reference_audio("char1") is False
        assert has_reference_audio("char2") is False
    
    def test_cleanup_all_clears_reference_audio_cache(self):
        """cleanup_all() clears ReferenceAudio._prompt_cache."""
        from lunavox_tts.Resources.Audio.ReferenceAudio import ReferenceAudio
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        # Verify cache access doesn't error and is clearable
        cache_before = len(ReferenceAudio._prompt_cache)
        runtime_manager.cleanup_all()
        cache_after = len(ReferenceAudio._prompt_cache)
        
        # After cleanup, cache should be empty
        assert cache_after == 0


class TestMultiCycleStability:
    """Tests for memory stability across multiple load/unload cycles."""
    
    @patch("lunavox_tts.API.characters.model_manager")
    def test_multi_cycle_state_empty(self, mock_mm):
        """After multiple load/unload cycles, API state should be empty."""
        from lunavox_tts.API.characters import unload_character
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio,
            clear_all_reference_audio,
            _reference_audios
        )
        
        clear_all_reference_audio()
        
        # Cycle 1
        set_reference_audio_config("cycle1_char", {'audio_path': 'c1.wav'})
        unload_character("cycle1_char")
        
        # Cycle 2
        set_reference_audio_config("cycle2_char", {'audio_path': 'c2.wav'})
        unload_character("cycle2_char")
        
        # Cycle 3
        set_reference_audio_config("cycle3_char", {'audio_path': 'c3.wav'})
        unload_character("cycle3_char")
        
        # Verify state is completely empty
        assert len(_reference_audios) == 0


class TestClearApiStateMethod:
    """Tests for RuntimeManager.clear_api_state specifically."""
    
    def test_clear_api_state_exists(self):
        """RuntimeManager has clear_api_state method."""
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        assert hasattr(runtime_manager, 'clear_api_state')
        assert callable(runtime_manager.clear_api_state)
    
    def test_try_empty_vram_exists(self):
        """RuntimeManager has try_empty_vram method."""
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        assert hasattr(runtime_manager, 'try_empty_vram')
        assert callable(runtime_manager.try_empty_vram)
    
    def test_clear_api_state_standalone(self):
        """clear_api_state can be called independently."""
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            has_reference_audio
        )
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        set_reference_audio_config("standalone_test", {'audio_path': 'test.wav'})
        
        runtime_manager.clear_api_state()
        
        assert has_reference_audio("standalone_test") is False
