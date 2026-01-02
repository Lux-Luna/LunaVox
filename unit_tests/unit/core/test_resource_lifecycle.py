"""
Unit tests for GlobalResourceManager and resource lifecycle.

Verifies that unload operations properly release singleton resources.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestGlobalResourceManager:
    """Tests for GlobalResourceManager singleton and unload operations."""
    
    def test_singleton_pattern(self):
        """GlobalResourceManager returns same instance."""
        from lunavox_tts.Utils.GlobalResourceManager import GlobalResourceManager
        
        mgr1 = GlobalResourceManager()
        mgr2 = GlobalResourceManager()
        
        assert mgr1 is mgr2
    
    def test_unload_hubert_sets_none(self):
        """unload_hubert sets ModelManager.cn_hubert to None."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        from lunavox_tts.ModelManager import model_manager
        
        # Set up a mock session
        model_manager.cn_hubert = MagicMock()
        assert model_manager.cn_hubert is not None
        
        # Unload
        global_resource_manager.unload_hubert()
        
        # Verify
        assert model_manager.cn_hubert is None
    
    def test_unload_zh_bert_sets_none(self):
        """unload_zh_bert sets ZhBert globals to None."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        from lunavox_tts.Languages.Chinese import ZhBert
        
        # Set up mock values
        ZhBert._ort_session = MagicMock()
        ZhBert._tokenizer = MagicMock()
        
        # Unload
        global_resource_manager.unload_zh_bert()
        
        # Verify
        assert ZhBert._ort_session is None
        assert ZhBert._tokenizer is None
    
    def test_unload_sv_sets_none(self):
        """unload_sv sets SpeakerVector globals to None."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        from lunavox_tts.Resources.Audio import SpeakerVector
        
        # Set up mock values
        SpeakerVector._sv_model = MagicMock()
        SpeakerVector._sv_model_path = "/path/to/model"
        
        # Unload
        global_resource_manager.unload_sv()
        
        # Verify
        assert SpeakerVector._sv_model is None
        assert SpeakerVector._sv_model_path is None
    
    def test_cleanup_all_clears_everything(self):
        """cleanup_all clears all resources."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        from lunavox_tts.ModelManager import model_manager
        from lunavox_tts.Languages.Chinese import ZhBert
        from lunavox_tts.Resources.Audio import SpeakerVector
        
        # Set up mock values
        model_manager.cn_hubert = MagicMock()
        ZhBert._ort_session = MagicMock()
        SpeakerVector._sv_model = MagicMock()
        
        # Cleanup all
        global_resource_manager.cleanup_all()
        
        # Verify all are None
        assert model_manager.cn_hubert is None
        assert ZhBert._ort_session is None
        assert SpeakerVector._sv_model is None
    
    def test_get_loaded_resources_returns_dict(self):
        """get_loaded_resources returns status dict."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        
        status = global_resource_manager.get_loaded_resources()
        
        assert isinstance(status, dict)
        assert "hubert" in status
        assert "zh_bert" in status
        assert "sv" in status
    
    def test_is_hubert_loaded_reflects_state(self):
        """is_hubert_loaded correctly reflects model state."""
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        from lunavox_tts.ModelManager import model_manager
        
        # Initially should be None/False
        model_manager.cn_hubert = None
        assert global_resource_manager.is_hubert_loaded() is False
        
        # After setting, should be True
        model_manager.cn_hubert = MagicMock()
        assert global_resource_manager.is_hubert_loaded() is True
        
        # Cleanup for next test
        model_manager.cn_hubert = None


class TestZhBertUnload:
    """Direct tests for ZhBert unload function."""
    
    def test_unload_model_function_exists(self):
        """ZhBert has unload_model function."""
        from lunavox_tts.Languages.Chinese.ZhBert import unload_model
        
        assert callable(unload_model)
    
    def test_unload_model_clears_session(self):
        """unload_model clears session and tokenizer."""
        from lunavox_tts.Languages.Chinese import ZhBert
        from lunavox_tts.Languages.Chinese.ZhBert import unload_model
        
        # Set mock values
        ZhBert._ort_session = MagicMock()
        ZhBert._tokenizer = MagicMock()
        
        # Unload
        unload_model()
        
        # Verify
        assert ZhBert._ort_session is None
        assert ZhBert._tokenizer is None


class TestSpeakerVectorUnload:
    """Direct tests for SpeakerVector unload function."""
    
    def test_unload_sv_model_function_exists(self):
        """SpeakerVector has unload_sv_model function."""
        from lunavox_tts.Resources.Audio.SpeakerVector import unload_sv_model
        
        assert callable(unload_sv_model)
    
    def test_unload_sv_model_clears_session(self):
        """unload_sv_model clears model and path."""
        from lunavox_tts.Resources.Audio import SpeakerVector
        from lunavox_tts.Resources.Audio.SpeakerVector import unload_sv_model
        
        # Set mock values
        SpeakerVector._sv_model = MagicMock()
        SpeakerVector._sv_model_path = "/some/path"
        
        # Unload
        unload_sv_model()
        
        # Verify
        assert SpeakerVector._sv_model is None
        assert SpeakerVector._sv_model_path is None
