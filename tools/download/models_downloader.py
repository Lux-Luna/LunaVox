import sys
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

def download_model(model_name: str, force: bool = False) -> Path:
    """Download a specific model from HF Hub."""
    if model_name not in REPO_MAP:
        valid = ", ".join(REPO_MAP.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {valid}")

    repo_id = REPO_MAP[model_name]
    print(f"[models_downloader] Starting download for {model_name} ({repo_id})...")
    
    # snapshot_download will automatically handle caching and skip if already exists
    path = snapshot_download(
        repo_id=repo_id,
        local_files_only=False,
        resume_download=True,
        # We don't force unless explicitly requested, HF hub handles mismatching blobs anyway
    )
    
    print(f"[models_downloader] Download complete: {path}")
    return Path(path)

def download_all():
    """Download all known models."""
    for name in REPO_MAP:
        try:
            download_model(name)
        except Exception as e:
            print(f"[models_downloader] Failed to download {name}: {e}")

if __name__ == "__main__":
    # If run directly, default to downloading a specific model or all
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(REPO_MAP.keys()) + ["all"], default="all")
    args = parser.parse_args()
    
    if args.model == "all":
        download_all()
    else:
        download_model(args.model)
