"""
Unit tests for API loading order and LoadingContext.

Verifies that strict-order loading works correctly for persona and reference modes.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock


class TestLoadingContext:
    """Tests for LoadingContext dataclass."""
    
    def test_create_for_persona_with_ge(self):
        """for_persona with global_emb creates persona_no_ge mode."""
        from lunavox_tts.Core.Model.LoadingContext import LoadingContext
        
        ctx = LoadingContext.for_persona(
            character_name="test",
            model_dir="/path/to/model",
            has_global_emb=True
        )
        
        assert ctx.mode == "persona_no_ge"
        assert ctx.skip_prompt_encoder is True
        assert "PROMPT_ENCODER" in ctx.get_skip_components()
    
    def test_create_for_persona_without_ge(self):
        """for_persona without global_emb creates persona_with_ge mode."""
        from lunavox_tts.Core.Model.LoadingContext import LoadingContext
        
        ctx = LoadingContext.for_persona(
            character_name="test",
            model_dir="/path/to/model",
            has_global_emb=False
        )
        
        assert ctx.mode == "persona_with_ge"
        assert ctx.skip_prompt_encoder is False
        assert len(ctx.get_skip_components()) == 0
    
    def test_create_for_reference(self):
        """for_reference creates reference mode with no skips."""
        from lunavox_tts.Core.Model.LoadingContext import LoadingContext
        
        ctx = LoadingContext.for_reference(
            character_name="test",
            model_dir="/path/to/model"
        )
        
        assert ctx.mode == "reference"
        assert ctx.skip_prompt_encoder is False
        assert len(ctx.get_skip_components()) == 0
    
    def test_get_skip_components_for_persona_no_ge(self):
        """persona_no_ge mode skips PROMPT_ENCODER."""
        from lunavox_tts.Core.Model.LoadingContext import LoadingContext
        
        ctx = LoadingContext(
            character_name="test",
            model_dir="/path",
            mode="persona_no_ge"
        )
        
        skip = ctx.get_skip_components()
        assert "PROMPT_ENCODER" in skip


class TestLoadPersonaOrder:
    """Tests for load_persona strict-order loading."""
    
    @patch("os.path.isdir", return_value=True)
    @patch("lunavox_tts.API.personas.persona_loader")
    @patch("lunavox_tts.API.personas.set_reference_audio_config")
    @patch("lunavox_tts.API.personas.load_character")
    @patch("lunavox_tts.API.personas.model_manager")
    @patch("lunavox_tts.API.personas.asset_manager")
    @patch("lunavox_tts.Utils.RuntimeManager.runtime_manager")
    def test_persona_with_ge_passes_skip_prompt_encoder(
        self, mock_rm, mock_am, mock_mm, mock_load_char, mock_set_ref, mock_loader, mock_isdir
    ):
        """load_persona with cached global_emb passes skip_prompt_encoder=True."""
        from lunavox_tts.API.personas import load_persona
        
        # Setup: persona has global_emb
        mock_ref = MagicMock()
        mock_ref.global_emb = [[0.1, 0.2]]  # Cached
        mock_ref.model_version = "v2ProPlus"
        mock_loader.return_value = mock_ref
        
        mock_mm.has_character.return_value = False
        mock_am.char_data_dir = MagicMock()
        mock_am.char_data_dir.__truediv__ = MagicMock(return_value=MagicMock())
        
        # Action
        load_persona("test_char", "/fake/persona/dir")
        
        # Verify load_character was called with skip_prompt_encoder=True
        mock_load_char.assert_called_once()
        call_kwargs = mock_load_char.call_args
        assert call_kwargs[1].get("skip_prompt_encoder") is True or call_kwargs[0][2] is True
    
    @patch("os.path.isdir", return_value=True)
    @patch("lunavox_tts.API.personas.persona_loader")
    @patch("lunavox_tts.API.personas.set_reference_audio_config")
    @patch("lunavox_tts.API.personas.load_character")
    @patch("lunavox_tts.API.personas.model_manager")
    @patch("lunavox_tts.API.personas.asset_manager")
    @patch("lunavox_tts.Utils.RuntimeManager.runtime_manager")
    def test_persona_without_ge_passes_skip_false(
        self, mock_rm, mock_am, mock_mm, mock_load_char, mock_set_ref, mock_loader, mock_isdir
    ):
        """load_persona without cached global_emb passes skip_prompt_encoder=False."""
        from lunavox_tts.API.personas import load_persona
        
        # Setup: persona has NO global_emb
        mock_ref = MagicMock()
        mock_ref.global_emb = None  # Not cached
        mock_ref.model_version = "v2"
        mock_loader.return_value = mock_ref
        
        mock_mm.has_character.return_value = False
        mock_am.char_data_dir = MagicMock()
        mock_am.char_data_dir.__truediv__ = MagicMock(return_value=MagicMock())
        
        # Action
        load_persona("test_char", "/fake/persona/dir")
        
        # Verify load_character was called with skip_prompt_encoder=False
        mock_load_char.assert_called_once()
        call_kwargs = mock_load_char.call_args
        # Either positional arg or keyword arg
        skip_value = call_kwargs[1].get("skip_prompt_encoder", call_kwargs[0][2] if len(call_kwargs[0]) > 2 else False)
        assert skip_value is False


class TestModelManagerSimplified:
    """Tests for simplified ModelManager without state healing."""
    
    def test_load_character_no_optimization_hint_check(self):
        """ModelManager.load_character no longer checks optimization hints."""
        from lunavox_tts.ModelManager import ModelManager
        
        # Verify the method signature doesn't reference get_optimization_hint
        import inspect
        source = inspect.getsource(ModelManager.load_character)
        
        assert "get_optimization_hint" not in source
        assert "set_optimization_hint" not in source


class TestRegistryNoHints:
    """Tests verifying registry no longer has hint system."""
    
    def test_registry_has_no_optimization_hint_methods(self):
        """ModelRegistry has no optimization hint methods."""
        from lunavox_tts.Core.Model.registry import ModelRegistry
        
        assert not hasattr(ModelRegistry, "set_optimization_hint") or \
               "set_optimization_hint" not in dir(ModelRegistry())
        assert not hasattr(ModelRegistry, "get_optimization_hint") or \
               "get_optimization_hint" not in dir(ModelRegistry())
    
    def test_model_entry_has_no_hint_field(self):
        """ModelEntry has no skip_prompt_encoder_hint field."""
        from lunavox_tts.Core.Model.registry import ModelEntry
        import dataclasses
        
        field_names = [f.name for f in dataclasses.fields(ModelEntry)]
        assert "skip_prompt_encoder_hint" not in field_names
