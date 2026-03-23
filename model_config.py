"""
LunaVox Multi-Model Configuration Center

Supported model variants:
  - base_small  : Qwen3-TTS-12Hz-0.6B-Base
  - custom_small: Qwen3-TTS-12Hz-0.6B-CustomVoice
  - base        : Qwen3-TTS-12Hz-1.7B-Base
  - custom      : Qwen3-TTS-12Hz-1.7B-CustomVoice
  - design      : Qwen3-TTS-12Hz-1.7B-VoiceDesign

Usage:
  1. Directly modify the `model = Models.xxx` line at the bottom of this file to switch models
  2. Or pass the --model <name> parameter via the command line
"""

from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

# Root directory for local HuggingFace hub cache
HF_HUB_ROOT = Path(HF_HUB_CACHE)

# Project root directory
REPO_ROOT = Path(__file__).resolve().parent


def get_snapshot(repo_name: str) -> Path:
    """Locates the actual snapshot path of the model in the HuggingFace cache (preferring official library for localization)"""
    repo_id = f"Qwen/{repo_name}"
    try:
        # Try to locate the local path using snapshot_download first
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception:
        # If the official library search fails or the snapshot is not downloaded, fall back to manual path joining logic
        snap_dir = HF_HUB_ROOT / f'models--Qwen--{repo_name}' / 'snapshots'
        if snap_dir.exists():
            snaps = [s for s in snap_dir.iterdir() if s.is_dir()]
            if snaps:
                return snaps[0]
        return HF_HUB_ROOT / f'models--Qwen--{repo_name}'


@dataclass
class ModelConfig:
    """Path configuration for a single model variant"""
    name: str       # Short identifier (used for CLI --model parameter)
    source: Path    # Original HF weight path
    dest: Path      # Output directory for conversion artifacts


class Models:
    """All available model variants"""
    base = ModelConfig(
        "base",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-Base'),
        REPO_ROOT / 'models' / 'base',
    )
    custom = ModelConfig(
        "custom",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-CustomVoice'),
        REPO_ROOT / 'models' / 'custom',
    )
    design = ModelConfig(
        "design",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-VoiceDesign'),
        REPO_ROOT / 'models' / 'design',
    )
    base_small = ModelConfig(
        "base_small",
        get_snapshot('Qwen3-TTS-12Hz-0.6B-Base'),
        REPO_ROOT / 'models' / 'base_small',
    )
    custom_small = ModelConfig(
        "custom_small",
        get_snapshot('Qwen3-TTS-12Hz-0.6B-CustomVoice'),
        REPO_ROOT / 'models' / 'custom_small',
    )

    ALL = [base, custom, design, base_small, custom_small]

    @classmethod
    def by_name(cls, name: str) -> ModelConfig:
        """Finds model configuration by name, raises an exception if it doesn't exist"""
        for m in cls.ALL:
            if m.name == name:
                return m
        valid = ', '.join(m.name for m in cls.ALL)
        raise ValueError(f"Unknown model '{name}'. Available: {valid}")


# ============================================================
# Currently selected model (modify this line directly or override via CLI --model)
# ============================================================
model = Models.base_small

MODEL_DIR = model.source
EXPORT_DIR = model.dest
