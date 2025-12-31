import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union

# Force logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to sys.path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from lunavox_tts.Utils.EnvManager import env_manager
from lunavox_tts.Audio.ReferenceAudio import ReferenceAudio
from lunavox_tts.Persona.PersonaManager import export_persona

class PersonaCreator:
    """
    High-quality Persona Solidification Tool.
    
    This tool extracts acoustic and speaker features from reference audio
    using CPU FP32 precision to ensure maximum quality and stability.
    """
    
    def __init__(self, model_version: str = "v2"):
        self.model_version = model_version
        
        # Force CPU mode for high-quality extraction
        logger.info("Initializing PersonaCreator in HIGH QUALITY CPU mode...")
        env_manager.set_mode("cpu")
        if not env_manager.ensure_environment():
            raise RuntimeError("Failed to set up CPU environment for Persona extraction.")

    def create(
        self,
        character_name: str,
        audio_path: str,
        text: Optional[str] = None,
        language: str = "auto",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Create a solidified persona from audio.
        
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
            root = Path(audio_path).parent.parent.parent.parent # Project Root maybe?
            # Better: use ResourceManager or relative to repo
            output_dir = os.path.join("CharacterData", "character", character_name)
            
        logger.info(f"--- Creating Persona: {character_name} ---")
        logger.info(f"Source: {audio_path}")
        logger.info(f"Lang: {language} | Version: {self.model_version}")

        # Extract features (this runs BERT, SSL, SV, Phonemizer)
        ref = ReferenceAudio(
            prompt_wav=audio_path,
            prompt_text=text,
            language=language,
            model_version=self.model_version
        )
        
        # Save to disk
        final_path = export_persona(
            ref_audio=ref,
            save_dir=output_dir,
            character_name=character_name,
            source_audio_path=audio_path
        )
        
        logger.info(f"鉁 Persona solidified successfully at: {final_path}")
        return final_path
