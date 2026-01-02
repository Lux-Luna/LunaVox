"""
Unit tests for EnvManager logic and environment switching.
"""
import pytest
from unittest.mock import MagicMock, patch
from lunavox_tts.Utils.EnvManager import EnvManager, EnvironmentStatus

class TestEnvLogic:
    def test_get_mode_respects_config(self, tmp_path):
        # Create a temp config file
        config_dir = tmp_path / "TTSData"
        config_dir.mkdir()
        config_file = config_dir / "env_config.json"
        config_file.write_text('{"mode": "gpu", "developer_mode": false}')
        
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em.config_file = config_file
            em._config = {"mode": "gpu"}
            em._mode_override = None
            
            assert em.get_mode() == "gpu"

    def test_get_mode_respects_override(self):
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em._config = {"mode": "cpu"}
            em._mode_override = "gpu"
            
            assert em.get_mode() == "gpu"

    @patch("onnxruntime.get_available_providers")
    @patch("importlib.metadata.distribution")
    def test_get_environment_status_gpu_ready(self, mock_dist, mock_providers):
        mock_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em._cached_env_status = None
            
            assert em.get_environment_status() == EnvironmentStatus.GPU_READY

    @patch("onnxruntime.get_available_providers")
    @patch("importlib.metadata.distribution")
    def test_get_environment_status_deps_missing(self, mock_dist, mock_providers):
        mock_providers.return_value = ["CPUExecutionProvider"]
        mock_dist.return_value = MagicMock() # Simulate onnxruntime-gpu is installed
        
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em._cached_env_status = None
            
            assert em.get_environment_status() == EnvironmentStatus.GPU_DEPS_MISSING

    @patch("onnxruntime.get_available_providers")
    @patch("importlib.metadata.distribution")
    def test_get_environment_status_cpu_only(self, mock_dist, mock_providers):
        mock_providers.return_value = ["CPUExecutionProvider"]
        mock_dist.side_effect = Exception("Not found")
        
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em._cached_env_status = None
            
            assert em.get_environment_status() == EnvironmentStatus.CPU_ONLY

    def test_ensure_environment_mismatch_raises(self):
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em.get_mode = MagicMock(return_value="gpu")
            em.get_environment_status = MagicMock(return_value=EnvironmentStatus.CPU_ONLY)
            em._print_gpu_instruction = MagicMock()
            
            from lunavox_tts.Core.Model.ExecutionPolicy import EnvironmentMismatchError
            with pytest.raises(EnvironmentMismatchError):
                em.ensure_environment()
            
            em._print_gpu_instruction.assert_called_once()

    def test_set_mode_warns_on_mismatch(self):
        with patch.object(EnvManager, '__init__', lambda x: None):
            em = EnvManager()
            em._config = {}
            em._save_config = MagicMock()
            em.get_environment_status = MagicMock(return_value=EnvironmentStatus.CPU_ONLY)
            em._print_gpu_instruction = MagicMock()
            
            em.set_mode("gpu")
            em._print_gpu_instruction.assert_called_once()
            assert em._config["mode"] == "gpu"
