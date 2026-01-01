"""
Pytest configuration and shared fixtures for LunaVox unit tests.

This file provides common fixtures and configuration for all unit tests.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_onnx_session():
    """Create a mock ONNX inference session."""
    session = MagicMock()
    session.run.return_value = [None]
    session.get_inputs.return_value = []
    session.get_outputs.return_value = []
    return session


@pytest.fixture
def mock_onnx_session_with_inputs():
    """Create a mock ONNX session that returns specific input names."""
    def factory(input_names: list):
        session = MagicMock()
        inputs = [MagicMock(name=name) for name in input_names]
        for inp, name in zip(inputs, input_names):
            inp.name = name
        session.get_inputs.return_value = inputs
        return session
    return factory


@pytest.fixture
def sample_reference_audio_config():
    """Sample reference audio configuration."""
    return {
        'audio_path': '/path/to/audio.wav',
        'audio_text': 'Hello world',
        'audio_lang': 'en',
        'model_version': 'v2',
        'is_persona': False,
    }


@pytest.fixture
def sample_persona_config():
    """Sample persona configuration."""
    return {
        'persona_dir': '/path/to/persona',
        'model_version': 'v2ProPlus',
        'is_persona': True,
        'prompt_audio': MagicMock(),
    }


# ============================================================================
# State Reset Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_state():
    """Reset API state before each test for isolation."""
    from lunavox_tts.API.state import clear_all_reference_audio
    clear_all_reference_audio()
    yield
    clear_all_reference_audio()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory with mock files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_info.json").write_text('{"version": "v2"}')
    return model_dir


@pytest.fixture
def temp_v2pp_model_dir(tmp_path):
    """Create a temporary v2ProPlus model directory."""
    model_dir = tmp_path / "model_v2pp"
    model_dir.mkdir()
    (model_dir / "model_info.json").write_text('{"version": "v2ProPlus"}')
    (model_dir / "prompt_encoder_fp32.onnx").write_bytes(b"mock")
    return model_dir
