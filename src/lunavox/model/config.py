from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

HF_HUB_ROOT = Path(HF_HUB_CACHE)

MODEL_REPOS: dict[str, str] = {
    "base": "Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base_small": "Qwen3-TTS-12Hz-0.6B-Base",
    "custom_small": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


def get_snapshot(repo_name: str) -> Path:
    repo_id = f"Qwen/{repo_name}"
    try:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception:
        snap_dir = HF_HUB_ROOT / f"models--Qwen--{repo_name}" / "snapshots"
        if snap_dir.exists():
            snaps = [s for s in snap_dir.iterdir() if s.is_dir()]
            if snaps:
                return snaps[0]
        return HF_HUB_ROOT / f"models--Qwen--{repo_name}"


@dataclass
class ModelConfig:
    name: str
    source: Path
    dest: Path


class Models:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._models = [
            ModelConfig(name, get_snapshot(repo), project_root / "models" / name)
            for name, repo in MODEL_REPOS.items()
        ]

    @property
    def all(self) -> list[ModelConfig]:
        return list(self._models)

    def by_name(self, name: str) -> ModelConfig:
        for model in self._models:
            if model.name == name:
                return model
        valid = ", ".join(m.name for m in self._models)
        raise ValueError(f"Unknown model '{name}'. Available: {valid}")

