"""
Unit tests for ModelManager simplified loading behavior.

After refactoring, state healing (optimization hints) has been removed.
Loading behavior is now determined by explicit skip_prompt_encoder parameter.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from lunavox_tts.ModelManager import ModelManager


class TestSimplifiedLoading:
    """Tests for simplified ModelManager without state healing."""
    
    @pytest.fixture
    def mm(self):
        with patch("lunavox_tts.ModelManager.monitor"):
            return ModelManager()

    def test_load_character_with_skip_prompt_encoder_true(self, mm, temp_v2pp_model_dir):
        """load_character with skip_prompt_encoder=True skips PROMPT_ENCODER."""
        char_name = "test_persona_mode"
        
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"T2S_ENCODER": Mock()}
            
            mm.load_character(char_name, str(temp_v2pp_model_dir), skip_prompt_encoder=True)
            
            args, kwargs = mock_loader.load_all.call_args
            assert "PROMPT_ENCODER" in kwargs.get("skip_components", set())

    def test_load_character_with_skip_prompt_encoder_false(self, mm, temp_v2pp_model_dir):
        """load_character with skip_prompt_encoder=False loads all components."""
        char_name = "test_reference_mode"
        
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"T2S_ENCODER": Mock(), "PROMPT_ENCODER": Mock()}
            
            mm.load_character(char_name, str(temp_v2pp_model_dir), skip_prompt_encoder=False)
            
            args, kwargs = mock_loader.load_all.call_args
            skip_components = kwargs.get("skip_components", set())
            assert "PROMPT_ENCODER" not in skip_components

    def test_load_character_same_path_returns_early(self, mm, temp_v2pp_model_dir):
        """load_character with same path returns early without reloading."""
        char_name = "test_cached"
        
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"T2S_ENCODER": Mock()}
            
            # First load
            mm.load_character(char_name, str(temp_v2pp_model_dir))
            first_call_count = mock_loader.load_all.call_count
            
            # Second load with same path
            mm.load_character(char_name, str(temp_v2pp_model_dir))
            second_call_count = mock_loader.load_all.call_count
            
            # Should not have loaded again
            assert second_call_count == first_call_count

    def test_cleanup_global_resources_delegates(self, mm):
        """cleanup_global_resources delegates to GlobalResourceManager."""
        with patch("lunavox_tts.Utils.GlobalResourceManager.global_resource_manager") as mock_grm:
            mm.cleanup_global_resources()
            mock_grm.cleanup_all.assert_called_once()


class TestNoStateHealing:
    """Verify state healing removal."""
    
    def test_no_optimization_hint_in_load_character(self):
        """ModelManager.load_character does not check optimization hints."""
        import inspect
        source = inspect.getsource(ModelManager.load_character)
        
        assert "get_optimization_hint" not in source
        assert "set_optimization_hint" not in source
        assert "Upgrading" not in source  # No upgrade logic
    
    def test_no_unload_prompt_encoder_method(self):
        """ModelManager no longer has unload_prompt_encoder method."""
        mm = ModelManager()
        assert not hasattr(mm, "unload_prompt_encoder")
    
    def test_no_unload_sv_model_method(self):
        """ModelManager no longer has unload_sv_model method."""
        mm = ModelManager()
        assert not hasattr(mm, "unload_sv_model")

