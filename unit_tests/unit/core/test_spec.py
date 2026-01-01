"""
Unit tests for lunavox_tts.Core.Model.spec module.

Tests ModelSpec, VocoderInputSpec, version detection, and input assembly.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock


class TestModelVersion:
    """Tests for ModelVersion enum."""
    
    def test_model_version_values(self):
        """Verify ModelVersion has correct values."""
        from lunavox_tts.Core.Model.spec import ModelVersion
        
        assert ModelVersion.V2.value == "v2"
        assert ModelVersion.V2_PRO.value == "v2Pro"
        assert ModelVersion.V2_PRO_PLUS.value == "v2ProPlus"
    
    def test_model_version_count(self):
        """Verify there are exactly 3 model versions."""
        from lunavox_tts.Core.Model.spec import ModelVersion
        
        assert len(ModelVersion) == 3


class TestGetModelSpec:
    """Tests for get_model_spec function."""
    
    def test_get_spec_v2(self):
        """get_model_spec('v2') returns V2_SPEC."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_SPEC
        
        spec = get_model_spec("v2")
        assert spec is V2_SPEC
    
    def test_get_spec_v2_default(self):
        """Unknown version defaults to v2."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_SPEC
        
        spec = get_model_spec("unknown")
        assert spec is V2_SPEC
    
    def test_get_spec_v2pro(self):
        """get_model_spec('v2Pro') returns V2_PRO_SPEC."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_PRO_SPEC
        
        spec = get_model_spec("v2Pro")
        assert spec is V2_PRO_SPEC
    
    def test_get_spec_v2proplus(self):
        """get_model_spec('v2ProPlus') returns V2_PRO_PLUS_SPEC."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_PRO_PLUS_SPEC
        
        spec = get_model_spec("v2ProPlus")
        assert spec is V2_PRO_PLUS_SPEC
    
    def test_get_spec_v2pp_alias(self):
        """get_model_spec('v2pp') returns V2_PRO_PLUS_SPEC."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_PRO_PLUS_SPEC
        
        spec = get_model_spec("v2pp")
        assert spec is V2_PRO_PLUS_SPEC
    
    def test_get_spec_case_insensitive(self):
        """Version string is case-insensitive."""
        from lunavox_tts.Core.Model.spec import get_model_spec, V2_PRO_PLUS_SPEC
        
        assert get_model_spec("V2PROPLUS") is V2_PRO_PLUS_SPEC
        assert get_model_spec("v2proplus") is V2_PRO_PLUS_SPEC


class TestVocoderInputSpec:
    """Tests for VocoderInputSpec and pre-defined specs."""
    
    def test_v2_required_inputs(self):
        """V2 vocoder requires text_seq, pred_semantic, ref_audio."""
        from lunavox_tts.Core.Model.spec import V2_VOCODER_INPUTS
        
        assert "text_seq" in V2_VOCODER_INPUTS.required_inputs
        assert "pred_semantic" in V2_VOCODER_INPUTS.required_inputs
        assert "ref_audio" in V2_VOCODER_INPUTS.required_inputs
    
    def test_v2pp_required_inputs(self):
        """V2ProPlus vocoder requires ge and ge_advanced."""
        from lunavox_tts.Core.Model.spec import V2_PRO_PLUS_VOCODER_INPUTS
        
        assert "ge" in V2_PRO_PLUS_VOCODER_INPUTS.required_inputs
        assert "ge_advanced" in V2_PRO_PLUS_VOCODER_INPUTS.required_inputs
    
    def test_v2pp_optional_inputs(self):
        """V2ProPlus has ref_audio as optional."""
        from lunavox_tts.Core.Model.spec import V2_PRO_PLUS_VOCODER_INPUTS
        
        assert "ref_audio" in V2_PRO_PLUS_VOCODER_INPUTS.optional_inputs
    
    def test_get_all_inputs(self):
        """get_all_inputs returns required + optional."""
        from lunavox_tts.Core.Model.spec import VocoderInputSpec
        
        spec = VocoderInputSpec(
            required_inputs=["a", "b"],
            optional_inputs=["c"]
        )
        all_inputs = spec.get_all_inputs()
        assert all_inputs == ["a", "b", "c"]


class TestDetectModelVersion:
    """Tests for detect_model_version function."""
    
    def test_detect_from_model_info_json(self, tmp_path):
        """Detect version from model_info.json."""
        from lunavox_tts.Core.Model.spec import detect_model_version
        import json
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_info.json").write_text(json.dumps({"version": "v2Pro"}))
        
        version = detect_model_version(str(model_dir))
        assert version == "v2Pro"
    
    def test_detect_from_prompt_encoder(self, tmp_path):
        """Detect v2ProPlus when prompt_encoder exists."""
        from lunavox_tts.Core.Model.spec import detect_model_version
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "prompt_encoder_fp32.onnx").write_bytes(b"mock")
        
        version = detect_model_version(str(model_dir))
        assert version == "v2ProPlus"
    
    def test_detect_from_dir_name_v2pp(self, tmp_path):
        """Detect v2ProPlus from directory name."""
        from lunavox_tts.Core.Model.spec import detect_model_version
        
        model_dir = tmp_path / "v2pp_model"
        model_dir.mkdir()
        
        version = detect_model_version(str(model_dir))
        assert version == "v2ProPlus"
    
    def test_detect_default_v2(self, tmp_path):
        """Default to v2 when no indicators."""
        from lunavox_tts.Core.Model.spec import detect_model_version
        
        model_dir = tmp_path / "some_model"
        model_dir.mkdir()
        
        version = detect_model_version(str(model_dir))
        assert version == "v2"


class TestAssembleVocoderInputs:
    """Tests for ModelSpec.assemble_vocoder_inputs."""
    
    def test_basic_inputs(self, mock_onnx_session_with_inputs):
        """Basic inputs are always included."""
        from lunavox_tts.Core.Model.spec import get_model_spec
        
        spec = get_model_spec("v2")
        mock_session = mock_onnx_session_with_inputs(["text_seq", "pred_semantic", "ref_audio"])
        
        # Create mock features
        features = MagicMock()
        features.audio_32k = np.zeros((16000,), dtype=np.float32)
        
        inputs = spec.assemble_vocoder_inputs(
            text_seq=np.array([[1, 2, 3]]),
            pred_semantic=np.array([[4, 5, 6]]),
            features=features,
            vocoder_session=mock_session,
        )
        
        assert "text_seq" in inputs
        assert "pred_semantic" in inputs
    
    def test_filters_unexpected_inputs(self, mock_onnx_session_with_inputs):
        """Inputs not expected by session are filtered out."""
        from lunavox_tts.Core.Model.spec import get_model_spec
        
        # Session only expects text_seq and pred_semantic (no ref_audio)
        spec = get_model_spec("v2")
        mock_session = mock_onnx_session_with_inputs(["text_seq", "pred_semantic"])
        
        features = MagicMock()
        features.audio_32k = np.zeros((16000,), dtype=np.float32)
        
        inputs = spec.assemble_vocoder_inputs(
            text_seq=np.array([[1, 2, 3]]),
            pred_semantic=np.array([[4, 5, 6]]),
            features=features,
            vocoder_session=mock_session,
        )
        
        # ref_audio should be filtered out since session doesn't expect it
        assert "ref_audio" not in inputs
    
    def test_v2pp_includes_global_emb(self, mock_onnx_session_with_inputs):
        """V2ProPlus includes ge and ge_advanced."""
        from lunavox_tts.Core.Model.spec import get_model_spec
        
        spec = get_model_spec("v2ProPlus")
        mock_session = mock_onnx_session_with_inputs([
            "text_seq", "pred_semantic", "ge", "ge_advanced"
        ])
        
        features = MagicMock()
        features.global_emb = np.zeros((1, 1024, 1), dtype=np.float32)
        features.global_emb_advanced = np.zeros((1, 1024, 1), dtype=np.float32)
        
        inputs = spec.assemble_vocoder_inputs(
            text_seq=np.array([[1, 2, 3]]),
            pred_semantic=np.array([[4, 5, 6]]),
            features=features,
            vocoder_session=mock_session,
        )
        
        assert "ge" in inputs
        assert "ge_advanced" in inputs
