import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Force logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to sys.path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from lunavox_tts.Utils.EnvManager import env_manager
from lunavox_tts.Resources.Audio.ReferenceAudio import ReferenceAudio
from lunavox_tts.Resources.Persona.PersonaManager import export_persona


class UniversalPersonaCreator:
    """
    Universal Persona Solidification Tool.
    
    Creates high-quality Universal Personas that work with both v2 and v2ProPlus models.
    
    Features extracted:
    - SSL content (HuBERT): Required for all versions
    - Speaker Vector (sv_emb): Required for v2Pro/v2ProPlus
    - Global Embeddings (global_emb): Required for v2ProPlus (computed via prompt_encoder)
    
    All features are extracted in CPU FP32 mode for maximum precision and stability.
    The resulting Persona can be used with v2 models (without prompt_encoder) or
    v2ProPlus models (without needing to load prompt_encoder at inference time).
    """
    
    def __init__(self):
        """Initialize the Universal Persona Creator in CPU FP32 mode."""
        logger.info("Initializing UniversalPersonaCreator in HIGH QUALITY CPU mode...")
        env_manager.set_mode("cpu")
        # Try to ensure environment, but continue if it fails due to just-installed runtime
        try:
            env_ok = env_manager.ensure_environment()
            if not env_ok:
                logger.warning("Environment setup returned False, but continuing anyway...")
        except Exception as e:
            logger.warning(f"Environment setup error: {e}, continuing anyway...")
        
        self._prompt_encoder = None
    
    def _ensure_prompt_encoder(self) -> None:
        """Load prompt_encoder model for global embedding computation."""
        if self._prompt_encoder is not None:
            return
        
        from lunavox_tts.Utils.ResourceManager import resource_manager
        from lunavox_tts.Core.Model.session import load_session_with_fp16_conversion, get_default_sess_options
        
        # Ensure v2pp pretrained models are available
        resource_manager.ensure_character_data(v2pp=True)
        
        model_dir = resource_manager.char_data_dir / "model" / "v2_pro_plus" / "pretrained"
        prompt_encoder_onnx = model_dir / "prompt_encoder_fp32.onnx"
        prompt_encoder_bin = model_dir / "prompt_encoder_fp16.bin"
        
        if not prompt_encoder_onnx.exists():
            raise FileNotFoundError(
                f"prompt_encoder ONNX not found at {prompt_encoder_onnx}. "
                f"Please ensure v2ProPlus pretrained models are downloaded."
            )
        
        logger.info(f"Loading prompt_encoder for global embedding extraction...")
        
        # Use FP16 patching method if bin file exists
        if prompt_encoder_bin.exists():
            self._prompt_encoder = load_session_with_fp16_conversion(
                str(prompt_encoder_onnx),
                str(prompt_encoder_bin),
                ["CPUExecutionProvider"],
                get_default_sess_options()
            )
        else:
            # Fallback to standard loading (unlikely to work with skeleton ONNX)
            import onnxruntime as ort
            sess_options = get_default_sess_options()
            self._prompt_encoder = ort.InferenceSession(
                str(prompt_encoder_onnx),
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
        
        logger.info("✓ prompt_encoder loaded successfully.")
    
    def create(
        self,
        character_name: str,
        audio_path: str,
        text: Optional[str] = None,
        language: str = "auto",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Create a Universal Persona from reference audio.
        
        The resulting Persona supports both v2 and v2ProPlus models.
        
        Args:
            character_name: Unique name for the persona.
            audio_path: Path to the reference .wav file.
            text: Transcript. If None, uses filename (stem).
            language: 'zh', 'en', 'ja', or 'auto'.
            output_dir: Where to save. Defaults to CharacterData/character/<character_name>.
            
        Returns:
            Absolute path to the created persona directory.
        """
        audio_path = os.fspath(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
            
        if text is None:
            text = Path(audio_path).stem
            logger.info(f"Using filename as prompt text: '{text}'")
            
        if output_dir is None:
            output_dir = os.path.join("CharacterData", "character", character_name)
            
        logger.info(f"--- Creating Universal Persona: {character_name} ---")
        logger.info(f"Source: {audio_path}")
        logger.info(f"Lang: {language}")

        # Ensure all extractor resources (HuBERT + SV + PromptEncoder)
        from lunavox_tts.Utils.ResourceManager import resource_manager
        resource_manager.ensure_extractor()
        self._ensure_prompt_encoder()

        # Extract base features (SSL, SV, Phonemes)
        logger.info("Extracting base features (SSL, SV, Phonemes)...")
        ref = ReferenceAudio(
            prompt_wav=audio_path,
            prompt_text=text,
            language=language,
            model_version="v2ProPlus"  # Request full feature extraction
        )
        
        # Run feature extraction pipeline to extract SSL, SV, and text features
        from lunavox_tts.Core.Processors.feature_extractor import feature_extractor
        feature_extractor.extract_all(ref, model_version="v2ProPlus")
        
        # Compute global embeddings for v2ProPlus compatibility
        logger.info("Computing global embeddings for v2ProPlus compatibility...")
        feature_extractor.extract_global_emb(ref, self._prompt_encoder)
        
        if ref.global_emb is None:
            logger.warning("Failed to compute global embeddings. Persona will only support v2 models.")
        else:
            logger.info(f"✓ Global embeddings computed: ge={ref.global_emb.shape}")
        
        # Save to disk
        final_path = export_persona(
            ref_audio=ref,
            save_dir=output_dir,
            character_name=character_name,
            source_audio_path=audio_path
        )
        
        logger.info(f"✓ Universal Persona solidified successfully at: {final_path}")
        return final_path


# Legacy alias for backward compatibility
PersonaCreator = UniversalPersonaCreator
