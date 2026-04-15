from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from lunavox.core.ui import console

from .config import get_model, model_keys


class ModelDownloader:
    """Download Qwen3-TTS source weights or pre-converted artifacts."""

    @staticmethod
    def download(model_name: str, force: bool = False) -> Path:
        spec = get_model(model_name)
        console.print(
            f"[info][model_manager:downloader] Starting download for "
            f"{spec.name} ({spec.repo_id})...[/info]"
        )
        path = snapshot_download(
            repo_id=spec.repo_id,
            local_files_only=False,
        )
        console.print(f"[info][model_manager:downloader] Download complete: {path}[/info]")
        return Path(path)

    @staticmethod
    def download_converted_model(model_name: str, project_root: Path) -> Path:
        """Download pre-converted GGUF/ONNX artifacts from the community repo."""
        get_model(model_name)  # validates name
        repo_id = "wkwong/Lunavox-Qwen3-TTS-GGUF"
        dest_dir = project_root / "models" / model_name
        console.print(
            f"[info]Pulling converted artifacts for [bold]'{model_name}'[/bold] "
            f"from {repo_id}...[/info]"
        )
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{model_name}/*"],
            local_dir=str(project_root / "models"),
        )
        console.print(f"[success]Pull complete: {dest_dir}[/success]")
        return dest_dir

    @staticmethod
    def download_all():
        for name in model_keys():
            try:
                ModelDownloader.download(name)
            except Exception as err:
                console.print(
                    f"[warning][model_manager:downloader] Failed to download {name}: {err}[/warning]"
                )
