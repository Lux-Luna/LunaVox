"""
Resource Manager - Modular, On-Demand Resource Fetching.

This module manages the downloading and verification of LunaVox resource packs.
Resources are partitioned into logical packs and fetched lazily based on usage.
"""
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Set

from huggingface_hub import snapshot_download
from .EnvManager import env_manager

logger = logging.getLogger(__name__)

REPO_ID = "wkwong/LunaVox"


class ResourcePack(Enum):
    """Available resource packs for on-demand loading."""
    BASE = "base"           # v2 model, EN G2P, EN Persona (~250 MB)
    CHINESE = "chinese"     # RoBERTa, CN G2P, CN Persona (~600 MB)
    JAPANESE = "japanese"   # JA Dict (pyopenjtalk), JA Persona (~50 MB)
    EXTRACTOR = "extractor" # HuBERT + SV models (~450 MB)
    V2PP = "v2pp"           # v2 Pro Plus model (~300 MB)


# Resource pack to HuggingFace patterns mapping
_PACK_PATTERNS = {
    ResourcePack.BASE: [
        "CharacterData/model/v2/pretrained/*",
        "CharacterData/character/luna_en/*",
        "CharacterData/audio/*",
        "TTSData/G2P/English/*",
    ],
    ResourcePack.CHINESE: [
        "RoBERTa/*",
        "TTSData/G2P/Chinese/*",
        "CharacterData/character/luna_zh/*",
    ],
    ResourcePack.JAPANESE: [
        "CharacterData/character/luna_ja/*",
        # Note: pyopenjtalk dict is bundled with the Python package
    ],
    ResourcePack.EXTRACTOR: [
        "TTSData/chinese-hubert-base/*",
        "TTSData/sv/*",
    ],
    ResourcePack.V2PP: [
        "CharacterData/model/v2_pro_plus/pretrained/*",
    ],
}


# Verification paths for each pack (at least one must exist to consider pack installed)
_PACK_VERIFICATION = {
    ResourcePack.BASE: [
        "CharacterData/model/v2/pretrained/vits_fp32.onnx",
        "TTSData/G2P/English/cmudict-fast.rep",
    ],
    ResourcePack.CHINESE: [
        "RoBERTa/RoBERTa.onnx",
    ],
    ResourcePack.JAPANESE: [
        "CharacterData/character/luna_ja/features.npz",
    ],
    ResourcePack.EXTRACTOR: [
        "TTSData/chinese-hubert-base/chinese-hubert-base.onnx",
        "TTSData/sv/eres2netv2.onnx",
    ],
    ResourcePack.V2PP: [
        "CharacterData/model/v2_pro_plus/pretrained/vits_fp32.onnx",
    ],
}


class ResourceManager:
    """
    Manages on-demand resource fetching from HuggingFace Hub.
    
    Resources are partitioned into logical packs (base, chinese, japanese, extractor, v2pp)
    and downloaded only when the corresponding feature is activated.
    """

    def __init__(self):
        self.repo_root = env_manager.repo_root
        self.tts_data_dir = self.repo_root / "TTSData"
        self.char_data_dir = self.repo_root / "CharacterData"
        self.roberta_dir = self.repo_root / "RoBERTa"
        self._loaded_packs: Set[ResourcePack] = set()
        self._check_existing_packs()

    def _check_existing_packs(self) -> None:
        """Initial scan to mark already-installed packs."""
        for pack in ResourcePack:
            if self._is_pack_installed(pack):
                self._loaded_packs.add(pack)
                logger.debug(f"Pack '{pack.value}' already installed.")

    def _is_pack_installed(self, pack: ResourcePack) -> bool:
        """Check if all verification files for a pack exist."""
        paths = _PACK_VERIFICATION.get(pack, [])
        if not paths:
            return True  # No verification needed
        return all((self.repo_root / p).exists() for p in paths)

    def ensure_pack(self, pack: ResourcePack) -> bool:
        """
        Ensure a resource pack is available. Downloads if missing.
        
        Args:
            pack: The resource pack to ensure.
            
        Returns:
            True if pack is available, False if download failed.
        """
        if pack in self._loaded_packs:
            return True
        
        if self._is_pack_installed(pack):
            self._loaded_packs.add(pack)
            return True
        
        logger.info(f"📦 Downloading resource pack: {pack.value}...")
        patterns = _PACK_PATTERNS.get(pack, [])
        if not patterns:
            logger.warning(f"No patterns defined for pack: {pack.value}")
            return False
        
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(self.repo_root),
                allow_patterns=patterns,
                local_dir_use_symlinks=False,
            )
            self._loaded_packs.add(pack)
            logger.info(f"✓ Resource pack '{pack.value}' installed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to download pack '{pack.value}': {e}")
            return False

    # ===== Convenience Methods (Backward Compatibility + Lazy Triggers) =====

    def ensure_base(self) -> bool:
        """Ensure base pack (v2 model, EN G2P) is available."""
        return self.ensure_pack(ResourcePack.BASE)

    def ensure_chinese(self) -> bool:
        """Ensure Chinese pack (RoBERTa, CN G2P) is available."""
        return self.ensure_pack(ResourcePack.CHINESE)

    def ensure_japanese(self) -> bool:
        """Ensure Japanese pack (JA resources) is available."""
        return self.ensure_pack(ResourcePack.JAPANESE)

    def ensure_extractor(self) -> bool:
        """Ensure feature extractor pack (HuBERT, SV) is available."""
        return self.ensure_pack(ResourcePack.EXTRACTOR)

    def ensure_v2pp(self) -> bool:
        """Ensure v2 Pro Plus model pack is available."""
        return self.ensure_pack(ResourcePack.V2PP)

    # ===== Legacy API (Deprecated, for backward compatibility) =====
    
    def ensure_tts_data(self, v2pp: bool = False) -> None:
        """[Deprecated] Use ensure_base() / ensure_extractor() instead."""
        self.ensure_base()
        if v2pp:
            self.ensure_extractor()
            self.ensure_v2pp()

    def ensure_character_data(self, v2pp: bool = False) -> None:
        """[Deprecated] Use ensure_base() / ensure_v2pp() instead."""
        self.ensure_base()
        if v2pp:
            self.ensure_v2pp()

    def ensure_roberta(self) -> None:
        """[Deprecated] Use ensure_chinese() instead."""
        self.ensure_chinese()


resource_manager = ResourceManager()
