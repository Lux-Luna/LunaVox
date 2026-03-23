from __future__ import annotations
from pathlib import Path
from huggingface_hub import snapshot_download

# Define model repo mapping (matching model_config.py's repo names)
REPO_MAP = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base_small": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "custom_small": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

class ModelDownloader:
    """Download a specific model from HF Hub."""
    @staticmethod
    def download(model_name: str, force: bool = False) -> Path:
        if model_name not in REPO_MAP:
            valid = ", ".join(REPO_MAP.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {valid}")

        repo_id = REPO_MAP[model_name]
        print(f"[model_manager:downloader] Starting download for {model_name} ({repo_id})...")
        
        path = snapshot_download(
            repo_id=repo_id,
            local_files_only=False,
            resume_download=True,
        )
        
        print(f"[model_manager:downloader] Download complete: {path}")
        return Path(path)

    @staticmethod
    def download_all():
        for name in REPO_MAP:
            try:
                ModelDownloader.download(name)
            except Exception as e:
                print(f"[model_manager:downloader] Failed to download {name}: {e}")
