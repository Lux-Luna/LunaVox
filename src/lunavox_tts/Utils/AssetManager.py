"""
Resource Manager - Modular, On-Demand Resource Fetching.

This module manages the downloading and verification of LunaVox resource packs.
Resources are partitioned into logical packs and fetched lazily based on usage.
"""
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Set, Optional, List

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
        "CharacterData/model/v2_pro_plus/pretrained/prompt_encoder_fp32.onnx",
    ],
}


# Resource pack to required Python packages mapping
_PACK_DEPENDENCIES = {
    ResourcePack.CHINESE: ["pypinyin", "cn2an", "jieba_fast", "g2pM"],
    ResourcePack.JAPANESE: ["pyopenjtalk-plus"],
}


class AssetManager:
    """
    Manages on-demand resource fetching from HuggingFace Hub.
    
    Resources are partitioned into logical packs (base, chinese, japanese, extractor, v2pp)
    and downloaded only when the corresponding feature is activated.
    """

    def __init__(self):
        self.repo_root = env_manager.repo_root
        self.data_root = env_manager.data_root
        self.tts_data_dir = self.data_root / "TTSData"
        self.char_data_dir = self.data_root / "CharacterData"
        self.roberta_dir = self.data_root / "RoBERTa"
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
        return all((self.data_root / p).exists() for p in paths)

    def ensure_pack(self, pack: ResourcePack, ignore_patterns: Optional[List[str]] = None) -> bool:
        """
        Ensure a resource pack is available. Downloads if missing.
        
        Args:
            pack: The resource pack to ensure.
            ignore_patterns: Optional list of glob patterns to ignore during download.
            
        Returns:
            True if pack is available, False if download failed.
        """
        # If no ignore patterns, we can use the cache
        if pack in self._loaded_packs and not ignore_patterns:
            return True
        
        # If already installed (all verification files exist), we're good
        if self._is_pack_installed(pack) and not ignore_patterns:
            self._loaded_packs.add(pack)
            # Still check dependencies even if files exist
            deps = _PACK_DEPENDENCIES.get(pack, [])
            if deps:
                from .DependencyManager import dependency_manager
                dependency_manager.check_dependencies(deps, pack.value.capitalize(), auto_install=True)
            return True
            
        # Optimization: If V2PP is requested with skip_prompt_encoder, and VITS is already here, 
        # we can skip the HF check to avoid latency.
        if pack == ResourcePack.V2PP and ignore_patterns == ["*prompt_encoder*"]:
            vits_path = self.char_data_dir / "model" / "v2_pro_plus" / "pretrained" / "vits_fp32.onnx"
            if vits_path.exists():
                logger.debug("V2PP VITS already exists, skipping partial HF pull.")
                return True
        
        logger.info(f"📦 Downloading resource pack: {pack.value}...")
        patterns = _PACK_PATTERNS.get(pack, [])
        if not patterns:
            logger.warning(f"No patterns defined for pack: {pack.value}")
            return False
        
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(self.data_root),
                allow_patterns=patterns,
                ignore_patterns=ignore_patterns,
                local_dir_use_symlinks=False,
            )
            # Only add to loaded_packs if we didn't use ignore_patterns (partial download)
            if not ignore_patterns:
                self._loaded_packs.add(pack)
            
            logger.info(f"✓ Resource pack '{pack.value}' processed.")
            
            # --- CHECK PYTHON DEPENDENCIES ---
            deps = _PACK_DEPENDENCIES.get(pack, [])
            if deps:
                from .DependencyManager import dependency_manager
                dependency_manager.check_dependencies(deps, pack.value.capitalize(), auto_install=True)
                
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

    def ensure_v2pp(self, skip_prompt_encoder: bool = False) -> bool:
        """Ensure v2 Pro Plus model pack is available."""
        ignore = ["*prompt_encoder*"] if skip_prompt_encoder else None
        return self.ensure_pack(ResourcePack.V2PP, ignore_patterns=ignore)

asset_manager = AssetManager()
