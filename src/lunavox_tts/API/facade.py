# Facade API for Easy Initialization
"""
Provides high-level, opinionated entry points for initializing LunaVox TTS.
Designed for tutorials and quick-start scripts.
"""

import logging
from pathlib import Path
from typing import Optional

from ..Utils.EnvManager import env_manager
from .characters import load_character
from .personas import load_persona

logger = logging.getLogger(__name__)

def initialize_tts(
    character_name: str, 
    setup_logging: bool = True,
    version: str = "v2",
    device: Optional[str] = None
) -> None:
    """
    High-level entry point to initialize the TTS system for a specific character.
    
    This function:
    1. Sets up basic logging (optional).
    2. Ensures the environment (CUDA, paths) is ready.
    3. Auto-detects proper Model and Persona paths based on standard structure.
    4. Loads the Persona (which auto-loads the Model).
    
    Args:
        character_name: The name of the character/persona to load (e.g., 'luna_en').
        setup_logging: If True, sets up basic logging if not already configured.
        version: Model version to use. Options: 'v2' (default), 'v2_pro_plus'.
        device: Device to use for inference. Options: 'cpu', 'gpu' (default depends on config).
    """
    # 1. Setup Logging
    if setup_logging:
        # Check if basic config is already done to avoid duplicates
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO, 
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    # 2. Ensure Environment
    if device:
        env_manager.set_mode(device)

    env_manager.ensure_environment()
    mode = env_manager.get_mode()
    logger.info(f"Initializing LunaVox TTS ({version}) in {mode.upper()} mode for: {character_name}")
    
    # 3. Path Resolution
    # Standard Structure:
    # lunavoxData/CharacterData/character/{name}
    data_root = env_manager.data_root
    char_data_root = data_root / "CharacterData"
    
    persona_path = char_data_root / "character" / character_name
    
    # 4. Load Logic
    if persona_path.exists():
        # load_persona internally handles model loading with optimization
        load_persona(character_name, str(persona_path), force_model_version=version)
    else:
        # Fallback: Just load the raw model if persona missing
        logger.warning(f"Persona directory '{character_name}' not found at {persona_path}. Loading raw model only.")
        
        if version == "v2_pro_plus":
             model_path = char_data_root / "model" / "v2_pro_plus" / "pretrained"
        else:
             model_path = char_data_root / "model" / "v2" / "pretrained"
             
        load_character(character_name, str(model_path))

