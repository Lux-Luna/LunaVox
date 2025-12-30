import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_english_g2p_dir() -> str:
    """Resolve path to LunaVox/TTSData/G2P/English"""
    from ..Utils.EnvManager import env_manager
    from ..Utils.ResourceManager import resource_manager
    resource_manager.ensure_tts_data()
    repo_root = env_manager.repo_root
    
    g2p_dir = repo_root / "TTSData" / "G2P" / "English"
    if not g2p_dir.exists():
        g2p_dir.mkdir(parents=True, exist_ok=True)
        
    return str(g2p_dir)

English_G2P_DIR = get_english_g2p_dir()

def ensure_english_g2p_resources():
    """Ensure essential English G2P resources exist."""
    required_files = [
        "checkpoint20.npz",
        "cmudict.rep",
        "wordsegment/unigrams.txt"
    ]
    
    for filename in required_files:
        file_path = os.path.join(English_G2P_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"English G2P resource {filename} not found at {English_G2P_DIR}.")
            logger.error(f"Please ensure English G2P data is copied to the TTSData/G2P/English directory.")
            # We don't raise here to allow the engine to fall back if possible, 
            # but usually it's fatal for the custom G2P.
            # raise FileNotFoundError(f"Missing English G2P resource: {file_path}")

