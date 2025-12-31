import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from .EnvManager import env_manager

logger = logging.getLogger(__name__)

REPO_ID = "wkwong/LunaVox"

class ResourceManager:
    def __init__(self):
        self.repo_root = env_manager.repo_root
        self.tts_data_dir = self.repo_root / "TTSData"
        self.char_data_dir = self.repo_root / "CharacterData"
        self.roberta_dir = self.repo_root / "RoBERTa"

    def ensure_tts_data(self, v2pp=False):
        """Ensure TTSData (G2P, hubert, and optionally sv) is present."""
        allow_patterns = ["TTSData/G2P/*", "TTSData/chinese-hubert-base/*"]
        if v2pp:
            allow_patterns.append("TTSData/sv/*")
        
        # Check if basic TTSData exists
        if not (self.tts_data_dir / "G2P").exists() or not (self.tts_data_dir / "chinese-hubert-base").exists():
            self._download(allow_patterns)
        elif v2pp and not (self.tts_data_dir / "sv").exists():
            self._download(["TTSData/sv/*"])

    def ensure_character_data(self, v2pp=False):
        """Ensure CharacterData (audio_resources and pretrained models) is present."""
        # 1. Always ensure audio_resources (per user request)
        if not (self.char_data_dir / "audio_resources").exists():
            self._download(["CharacterData/audio_resources/*"])
        
        # 2. Version-specific pretrained model pulling
        if v2pp:
            # v2ProPlus check
            if not (self.char_data_dir / "character_model" / "v2_pro_plus" / "pretrained").exists():
                self._download(["CharacterData/character_model/v2_pro_plus/pretrained/*"])
        else:
            # Standard v2 check
            if not (self.char_data_dir / "character_model" / "v2" / "pretrained").exists():
                self._download(["CharacterData/character_model/v2/pretrained/*"])

    def ensure_roberta(self):
        """Ensure RoBERTa model is present (for Chinese TTS)."""
        if not self.roberta_dir.exists() or not any(self.roberta_dir.iterdir()):
            self._download(["RoBERTa/*"])

    def _download(self, allow_patterns):
        logger.info(f"Downloading required resources from HF: {allow_patterns}")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(self.repo_root),
                allow_patterns=allow_patterns,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            logger.error(f"Failed to download resources: {e}")

resource_manager = ResourceManager()





