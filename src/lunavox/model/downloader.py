from __future__ import annotations
from pathlib import Path
from huggingface_hub import snapshot_download
from .config import MODEL_REPOS

REPO_MAP = {name: f"Qwen/{repo}" for name, repo in MODEL_REPOS.items()}

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
    def download_converted_model(model_name: str, project_root: Path) -> Path:
        """Download pre-converted GGUF/ONNX artifacts from the community repo."""
        repo_id = "wkwong/Lunavox-Qwen3-TTS-GGUF"
        dest_dir = project_root / "models" / model_name
        
        print(f"[model_manager:downloader] Pulling converted artifacts for '{model_name}' from {repo_id}...")
        
        # We download to the models/<name> directory
        # Using allow_patterns to only get the specific model's folder if possible, 
        # but snapshot_download handles local_dir well.
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{model_name}/*"],
            local_dir=str(project_root / "models"),
            local_dir_use_symlinks=False,
        )
        
        print(f"[model_manager:downloader] Pull complete: {dest_dir}")
        return dest_dir

    @staticmethod
    def download_all():
        for name in REPO_MAP:
            try:
                ModelDownloader.download(name)
            except Exception as e:
                print(f"[model_manager:downloader] Failed to download {name}: {e}")
