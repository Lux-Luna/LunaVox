
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to include src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from lunavox_tts.Utils.RuntimeManager import runtime_manager
from lunavox_tts.Languages.Chinese import ZhBert
from lunavox_tts.Resources.Audio import SpeakerVector
from lunavox_tts.ModelManager import model_manager

def test_zh_bert_cleanup():
    # Setup: simulate loaded state
    ZhBert._ort_session = MagicMock()
    ZhBert._tokenizer = MagicMock()
    
    assert ZhBert.is_loaded() is True
    
    # Action
    runtime_manager.unload_zh_bert()
    
    # Verify
    assert ZhBert.is_loaded() is False
    assert ZhBert._ort_session is None
    assert ZhBert._tokenizer is None

def test_sv_cleanup():
    # Setup: simulate loaded state
    SpeakerVector._sv_model = MagicMock()
    
    assert SpeakerVector.is_loaded() is True
    
    # Action
    runtime_manager.unload_sv()
    
    # Verify
    assert SpeakerVector.is_loaded() is False
    assert SpeakerVector._sv_model is None

def test_hubert_cleanup():
    # Setup: simulate loaded state
    from lunavox_tts.ResourceManager import resource_manager
    resource_manager.cn_hubert = MagicMock()
    
    # Action
    runtime_manager.unload_hubert()
    
    # Verify
    assert resource_manager.cn_hubert is None

def test_cleanup_all():
    # Setup all
    ZhBert._ort_session = MagicMock()
    SpeakerVector._sv_model = MagicMock()
    
    from lunavox_tts.ResourceManager import resource_manager
    resource_manager.cn_hubert = MagicMock()
    
    # Action
    runtime_manager.cleanup_all()
    
    # Verify all
    assert ZhBert.is_loaded() is False
    assert SpeakerVector.is_loaded() is False
    assert resource_manager.cn_hubert is None

if __name__ == "__main__":
    pytest.main([__file__])
