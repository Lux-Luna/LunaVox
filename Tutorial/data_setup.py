import os
import zipfile
import io
import shutil
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import snapshot_download
import requests


REPO_ID = "wkwong/LunaVox"

REPO_ROOT = Path(__file__).parent.parent  # Go up one level from Tutorial to repo root
DATA_DIR = REPO_ROOT / "Data"
CHAR_DIR = DATA_DIR / "character_model"
AUDIO_DIR = DATA_DIR / "audio_resources"
AUDIO_LANGUAGE_FOLDERS = ["Chinese", "English", "Japanese"]

REQUIRED_CN_HUBERT = DATA_DIR / "chinese-hubert-base.onnx"
REQUIRED_OPENJTALK_DIR = DATA_DIR / "open_jtalk_dic_utf_8-1.11"
REQUIRED_CHINESE_ROBERTA_DIR = DATA_DIR / "chinese-roberta-wwm-ext-large"

CHAR_REQUIRED_FILES = [
    "t2s_encoder_fp32.onnx",
    "t2s_first_stage_decoder_fp32.onnx",
    "t2s_stage_decoder_fp32.onnx",
    "t2s_shared_fp16.bin",
    "vits_fp32.onnx",
    "vits_fp16.bin",
]

def _copy_missing(src: Path, dst: Path) -> None:
    """Recursively copy files/dirs from src to dst, skipping paths that already exist locally."""
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())


def list_local_model_dirs() -> List[Path]:
    """Return character model leaf directories, e.g. Data/character_model/v2/pretrained."""
    results: List[Path] = []
    if not CHAR_DIR.exists():
        return results
    for family_dir in CHAR_DIR.iterdir():
        if not family_dir.is_dir():
            continue
        for model_dir in family_dir.iterdir():
            if model_dir.is_dir():
                results.append(model_dir)
    return sorted(results, key=lambda p: str(p.relative_to(REPO_ROOT)))


def list_existing_characters() -> List[Path]:
    if not CHAR_DIR.exists():
        return []
    # Return all immediate subdirectories under character_model as existing characters
    candidates = [p for p in CHAR_DIR.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.name)
    return candidates


def list_existing_audio_characters() -> List[Path]:
    """
    Returns audio preset directories following the language-based layout.

    Falls back to any subdirectories if explicit language folders are absent
    to maintain backward compatibility.
    """
    if not AUDIO_DIR.exists():
        return []
    language_dirs: List[Path] = []
    for lang in AUDIO_LANGUAGE_FOLDERS:
        lang_dir = AUDIO_DIR / lang
        if lang_dir.is_dir():
            language_dirs.append(lang_dir)
    if language_dirs:
        language_dirs.sort(key=lambda p: p.name.lower())
        return language_dirs
    fallback = [p for p in AUDIO_DIR.iterdir() if p.is_dir()]
    fallback.sort(key=lambda p: p.name.lower())
    return fallback


def audio_language_missing_items() -> List[str]:
    """
    Return detailed missing info for language audio resources.
    """
    missing: List[str] = []
    for lang in AUDIO_LANGUAGE_FOLDERS:
        lang_dir = AUDIO_DIR / lang
        rel_dir = str(lang_dir.relative_to(REPO_ROOT)) + "/"
        if not lang_dir.exists():
            missing.append(rel_dir)
            continue
        if not any(p.is_file() and p.suffix.lower() == ".wav" for p in lang_dir.iterdir()):
            missing.append(f"{rel_dir}(missing .wav files)")
    return missing


def character_missing_files(char_path: Path) -> List[str]:
    missing: List[str] = []
    for name in CHAR_REQUIRED_FILES:
        if not (char_path / name).exists():
            missing.append(name)
    return missing


def need_download() -> Tuple[bool, List[Tuple[str, List[str]]]]:
    missing_summary: List[Tuple[str, List[str]]] = []

    base_missing: List[str] = []
    if not REQUIRED_CN_HUBERT.exists():
        base_missing.append(str(REQUIRED_CN_HUBERT.relative_to(REPO_ROOT)))
    if not REQUIRED_OPENJTALK_DIR.exists():
        base_missing.append(str(REQUIRED_OPENJTALK_DIR.relative_to(REPO_ROOT)) + "/")
    if not REQUIRED_CHINESE_ROBERTA_DIR.exists():
        base_missing.append(str(REQUIRED_CHINESE_ROBERTA_DIR.relative_to(REPO_ROOT)) + "/")
    if base_missing:
        missing_summary.append(("base", base_missing))

    # Check character models according to local structure
    if not CHAR_DIR.exists() or not any(p.is_dir() for p in CHAR_DIR.iterdir()):
        missing_summary.append(("character_model", [str(CHAR_DIR.relative_to(REPO_ROOT)) + "/"]))
    else:
        # Inspect each local model dir and record missing files
        for model_dir in list_local_model_dirs():
            missing_files = character_missing_files(model_dir)
            if missing_files:
                # Report missing file paths relative to repo root
                rel_missing = [str((model_dir / name).relative_to(REPO_ROOT)) for name in missing_files]
                scope = f"character_files:{str(model_dir.relative_to(REPO_ROOT))}"
                missing_summary.append((scope, rel_missing))

    # Check audio resources
    audio_missing = audio_language_missing_items()
    if audio_missing:
        missing_summary.append(("audio_resources", audio_missing))

    return (len(missing_summary) > 0), missing_summary


def ensure_data_from_hf() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    is_missing, items = need_download()
    if not is_missing:
        print("All required Data dependencies are present.")
        return

    print("Some Data dependencies are missing:")
    for scope, names in items:
        print(f"- {scope}: {', '.join(names)}")

    # Optional token support for private repos
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    print(f"Downloading missing assets from Hugging Face repo: {REPO_ID} ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        local_dir=None,
        local_dir_use_symlinks=False,
        token=hf_token,
        allow_patterns=["Data/**"]
    )
    hf_root = Path(local_dir)

    src_cn = hf_root / "Data" / "chinese-hubert-base.onnx"
    if src_cn.exists() and not REQUIRED_CN_HUBERT.exists():
        REQUIRED_CN_HUBERT.parent.mkdir(parents=True, exist_ok=True)
        REQUIRED_CN_HUBERT.write_bytes(src_cn.read_bytes())

    src_dict = hf_root / "Data" / "open_jtalk_dic_utf_8-1.11"
    if src_dict.exists() and not REQUIRED_OPENJTALK_DIR.exists():
        REQUIRED_OPENJTALK_DIR.mkdir(parents=True, exist_ok=True)
        for path in src_dict.rglob("*"):
            if path.is_dir():
                (REQUIRED_OPENJTALK_DIR / path.relative_to(src_dict)).mkdir(parents=True, exist_ok=True)
            else:
                dst = REQUIRED_OPENJTALK_DIR / path.relative_to(src_dict)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(path.read_bytes())

    # Download Chinese RoBERTa model if missing
    if not REQUIRED_CHINESE_ROBERTA_DIR.exists():
        print("Downloading Chinese RoBERTa model from hfl/chinese-roberta-wwm-ext-large...")
        roberta_local_dir = snapshot_download(
            repo_id="hfl/chinese-roberta-wwm-ext-large",
            local_dir=str(REQUIRED_CHINESE_ROBERTA_DIR),
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print(f"Chinese RoBERTa model downloaded to: {roberta_local_dir}")

    char_src_root = hf_root / "Data" / "character_model"
    if char_src_root.exists():
        CHAR_DIR.mkdir(parents=True, exist_ok=True)
        local_chars = [p for p in CHAR_DIR.iterdir() if p.is_dir()]
        if not local_chars:
            # No local characters; copy entire remote character_model
            _copy_missing(char_src_root, CHAR_DIR)
        else:
            # Complement existing local characters only
            for local_char in local_chars:
                src_char = char_src_root / local_char.name
                if src_char.exists() and src_char.is_dir():
                    _copy_missing(src_char, local_char)

    # Download/complement audio resources
    audio_src_root = hf_root / "Data" / "audio_resources"
    if audio_src_root.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        _copy_missing(audio_src_root, AUDIO_DIR)

    print("Data setup completed.")


if __name__ == "__main__":
    ensure_data_from_hf()


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "CHAR_DIR",
    "AUDIO_DIR",
    "AUDIO_LANGUAGE_FOLDERS",
    "REQUIRED_CN_HUBERT",
    "REQUIRED_OPENJTALK_DIR",
    "REQUIRED_CHINESE_ROBERTA_DIR",
    "CHAR_REQUIRED_FILES",
    "list_existing_characters",
    "list_existing_audio_characters",
    "audio_language_missing_items",
    "character_missing_files",
    "need_download",
    "ensure_data_from_hf",
]


