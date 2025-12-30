import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_chinese_g2p_dir() -> str:
    # Resolve path to LunaVox/TTSData/G2P/Chinese
    # Current file: src/lunavox_tts/Chinese/Resources.py
    # Root: src/lunavox_tts/../../ (LunaVox root) -> Data
    
    from ..Utils.EnvManager import env_manager
    from ..Utils.ResourceManager import resource_manager
    resource_manager.ensure_tts_data()
    repo_root = env_manager.repo_root
    
    g2p_dir = repo_root / "TTSData" / "G2P" / "Chinese"
    if not g2p_dir.exists():
        g2p_dir.mkdir(parents=True, exist_ok=True)
        
    return str(g2p_dir)

Chinese_G2P_DIR = get_chinese_g2p_dir()

def ensure_g2p_resources():
    """Ensure polyphonic.pickle and opencpop-strict.txt exist."""
    files = ["polyphonic.pickle", "opencpop-strict.txt"]
    
    for filename in files:
        file_path = os.path.join(Chinese_G2P_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"Chinese G2P resource {filename} not found at {Chinese_G2P_DIR}.")
            logger.error(f"Please ensure {filename} is present in the G2P directory.")
            raise FileNotFoundError(f"Missing G2P resource: {file_path}")

