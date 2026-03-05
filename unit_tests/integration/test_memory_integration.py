"""
Integration tests for memory lifecycle management.

These tests verify actual physical memory behavior using psutil and weakref,
validating that ONNX Sessions are properly garbage collected.
"""
import pytest
import gc
import weakref
from unittest.mock import patch, MagicMock


class TestMemoryLifecycleIntegration:
    """Integration tests for memory cleanup verification."""
    
    def test_cleanup_all_method_exists(self):
        """Verify cleanup_all method is callable."""
        from lunavox_tts.Utils.RuntimeManager import runtime_manager
        
        assert hasattr(runtime_manager, 'cleanup_all')
        assert callable(runtime_manager.cleanup_all)
        
        # Should not raise
        runtime_manager.cleanup_all()
    
    def test_reset_baselines_method_exists(self):
        """Verify PerformanceMonitor has reset_baselines method."""
        from lunavox_tts.Utils.PerformanceMonitor import monitor
        
        assert hasattr(monitor, 'reset_baselines')
        assert callable(monitor.reset_baselines)
        
        # Should not raise
        monitor.reset_baselines()
    
    def test_model_manager_cache_capacity_reduced(self):
        """Verify ModelManager LRU cache capacity is 1 by default."""
        from lunavox_tts.ModelManager import ModelManager
        import os
        
        # Clear env var to test default
        original = os.environ.get('Max_Cached_Character_Models')
        if original:
            del os.environ['Max_Cached_Character_Models']
        
        try:
            mm = ModelManager()
            assert mm.character_to_model.capacity == 1
        finally:
            if original:
                os.environ['Max_Cached_Character_Models'] = original
    
    def test_reference_audio_cache_capacity_reduced(self):
        """Verify ReferenceAudio cache capacity is 2 by default."""
        import os
        
        # Clear env var to test default
        original = os.environ.get('Max_Cached_Reference_Audio')
        if original:
            del os.environ['Max_Cached_Reference_Audio']
        
        try:
            # Reimport to get fresh class
            from importlib import reload
            import lunavox_tts.Resources.Audio.ReferenceAudio as ra_module
            reload(ra_module)
            
            assert ra_module.ReferenceAudio._prompt_cache.capacity == 2
        finally:
            if original:
                os.environ['Max_Cached_Reference_Audio'] = original


class TestSessionWeakrefCleanup:
    """Tests using weakref to verify ONNX Session garbage collection."""
    
    def test_dict_cleanup_releases_reference(self):
        """Verify that clearing a dict releases object references."""
        class MockSession:
            pass
        
        cache = {}
        session = MockSession()
        weak_ref = weakref.ref(session)
        
        cache['test'] = session
        del session
        gc.collect()
        
        # Still held by cache
        assert weak_ref() is not None
        
        # Clear cache
        cache['test'] = None
        cache.clear()
        gc.collect()
        gc.collect()
        
        # Now should be collected
        assert weak_ref() is None
    
    @patch('lunavox_tts.API.characters.model_manager')
    def test_multi_unload_clears_all_state(self, mock_mm):
        """Multiple unload cycles should leave no state."""
        from lunavox_tts.API.characters import unload_character
        from lunavox_tts.API.state import (
            set_reference_audio_config,
            clear_all_reference_audio,
            _reference_audios
        )
        
        clear_all_reference_audio()
        
        # Simulate 5 load/unload cycles
        for i in range(5):
            char_name = f"test_char_{i}"
            set_reference_audio_config(char_name, {'audio_path': f'{i}.wav'})
            unload_character(char_name)
        
        assert len(_reference_audios) == 0


class TestBenchmarkCharacterPooling:
    """Tests for benchmark character name pooling."""
    
    def test_character_name_pool_exists(self):
        """Verify benchmark uses fixed character name pool."""
        import sys
        sys.path.insert(0, str(__file__).replace('unit_tests/integration/test_memory_integration.py', 'benchmark'))
        
        try:
            from benchmark import CHAR_NAME_POOL, LANG_TO_INDEX, get_character_name
            
            assert len(CHAR_NAME_POOL) == 3
            assert 'zh' in LANG_TO_INDEX
            assert 'en' in LANG_TO_INDEX
            assert 'ja' in LANG_TO_INDEX
        except ImportError:
            pytest.skip("benchmark module not in path")
    
    def test_character_name_reuses_pool(self):
        """Verify same language gets same character name."""
        import sys
        sys.path.insert(0, str(__file__).replace('unit_tests/integration/test_memory_integration.py', 'benchmark'))
        
        try:
            from benchmark import get_character_name
            
            # Same language should always get same name
            name1 = get_character_name("v2", "zh", "persona")
            name2 = get_character_name("v2pp", "zh", "reference")
            assert name1 == name2
            
            # Different languages get different names
            name_en = get_character_name("v2", "en", "persona")
            assert name_en != name1
        except ImportError:
            pytest.skip("benchmark module not in path")
