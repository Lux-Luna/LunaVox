"""
Unit tests for ModelManager State Healing logic.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from lunavox_tts.ModelManager import ModelManager
from lunavox_tts.Core.Model.registry import model_registry

class TestStateHealing:
    @pytest.fixture
    def mm(self):
        with patch("lunavox_tts.ModelManager.monitor"):
            return ModelManager()

    def test_load_character_auto_skips_pe_with_hint(self, mm, temp_v2pp_model_dir):
        """Test that load_character skips PE if registry has an optimization hint."""
        char_name = "test_luna_skip"
        
        # 1. Set the optimization hint in the registry
        model_registry.set_optimization_hint(char_name, skip_prompt_encoder=True)
        
        # 2. Mock the model loader to see what components it gets
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"T2S_ENCODER": Mock()}
            
            # 3. Call load_character WITHOUT explicitly passing skip_prompt_encoder=True
            mm.load_character(char_name, str(temp_v2pp_model_dir))
            
            # 4. Verify that skip_components in load_all included PROMPT_ENCODER
            args, kwargs = mock_loader.load_all.call_args
            assert "PROMPT_ENCODER" in kwargs.get("skip_components", set())

    @patch("lunavox_tts.API.personas.set_reference_audio_config")
    @patch("lunavox_tts.API.personas.persona_loader")
    def test_load_persona_unloads_pe_if_already_loaded(self, mock_loader, mock_state, mm, tmp_path):
        """Test that load_persona unloads PE if the model was already loaded with it."""
        from lunavox_tts.API.personas import load_persona
        char_name = "test_luna_unload"
        persona_dir = tmp_path / "persona"
        persona_dir.mkdir()
        
        # Setup mock persona with cached embeddings
        mock_ref = MagicMock()
        mock_ref.global_emb = [0.1, 0.2] # Cached
        mock_loader.return_value = mock_ref
        
        # Mock ModelManager instance methods
        mm.has_character = MagicMock(return_value=True)
        mm.get_character_version = MagicMock(return_value="v2ProPlus")
        mm.unload_prompt_encoder = MagicMock()
        
        with patch("lunavox_tts.API.personas.model_manager", mm):
            # 1. Call load_persona
            load_persona(char_name, str(persona_dir))
            
            # 2. Verify Registry hint was set
            assert model_registry.get_optimization_hint(char_name) is True
            
            # 3. Verify ModelManager.unload_prompt_encoder was called
            mm.unload_prompt_encoder.assert_called_with(char_name)

    def test_load_character_upgrades_from_persona_to_reference(self, mm, temp_v2pp_model_dir):
        """Test that calling load_character without skip_prompt_encoder loads missing PE."""
        char_name = "test_luna_upgrade"
        
        # 1. Initial load WITH skip hint (Persona mode)
        model_registry.set_optimization_hint(char_name, skip_prompt_encoder=True)
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"T2S_ENCODER": Mock()}
            mm.load_character(char_name, str(temp_v2pp_model_dir))
            
            # Simulate entry in mm.character_to_model that lacks PE
            mm.character_to_model[char_name] = {"T2S_ENCODER": Mock(), "PROMPT_ENCODER": None}
        
        # 2. Reset hint (Reference mode requested by user)
        model_registry.set_optimization_hint(char_name, skip_prompt_encoder=False)
        
        # 3. Call load_character again (should trigger "Upgrading" logic)
        with patch("lunavox_tts.ModelManager.model_loader") as mock_loader:
            mock_loader.load_all.return_value = {"PROMPT_ENCODER": Mock()}
            mm.load_character(char_name, str(temp_v2pp_model_dir))
            
            # Verify load_all was called
            assert mock_loader.load_all.called
            # Verify skip_components did NOT include PROMPT_ENCODER this time
            args, kwargs = mock_loader.load_all.call_args
            assert "PROMPT_ENCODER" not in kwargs.get("skip_components", set())
