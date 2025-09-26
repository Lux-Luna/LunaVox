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
TEXT_DIR = REPO_ROOT / "src" / "text"

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


def list_existing_characters() -> List[Path]:
    if not CHAR_DIR.exists():
        return []
    return [p for p in CHAR_DIR.iterdir() if p.is_dir()]


def list_existing_audio_characters() -> List[Path]:
    if not AUDIO_DIR.exists():
        return []
    return [p for p in AUDIO_DIR.iterdir() if p.is_dir()]


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

    # Check character models
    existing_chars = list_existing_characters()
    if not existing_chars:
        missing_summary.append(("character_any", CHAR_REQUIRED_FILES.copy()))
    else:
        first_char = existing_chars[0]
        m = character_missing_files(first_char)
        if m:
            missing_summary.append((first_char.name, m))

    # Check audio resources
    existing_audio_chars = list_existing_audio_characters()
    if not existing_audio_chars:
        missing_summary.append(("audio_resources", ["audio_resources directory with character folders"]))

    return (len(missing_summary) > 0), missing_summary


def ensure_g2pw_model() -> None:
    """Ensure LunaVox/src/text/G2PWModel exists; if missing, download from ModelScope and unzip.

    This mirrors the runtime logic found in GPT-SoVITS text.g2pw.onnx_api but targets LunaVox/src/text.
    """
    target_dir = TEXT_DIR / "G2PWModel"
    if target_dir.exists():
        return

    parent_directory = target_dir.parent
    parent_directory.mkdir(parents=True, exist_ok=True)

    # Prefer local copy from GPT-SoVITS if available
    local_source_dir = REPO_ROOT / "GPT-SoVITS" / "GPT_SoVITS" / "text" / "G2PWModel"
    if local_source_dir.exists():
        try:
            print(f"Copying G2PWModel from {local_source_dir} ...")
            shutil.copytree(str(local_source_dir), str(target_dir))
            print(f"G2PWModel ready at: {target_dir}")
            return
        except Exception as e:
            print(f"Copy G2PWModel failed, fallback to download. Reason: {e}")

    # Fallback to download from ModelScope (no zip saved to disk)
    extract_dir = parent_directory / "G2PWModel_1.1"
    final_dir = parent_directory / "G2PWModel"
    url = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"

    try:
        print("G2PWModel not found locally. Downloading from ModelScope ...")
        buf = io.BytesIO()
        with requests.get(url, stream=True, timeout=60) as r:  # type: ignore[arg-type]
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buf.write(chunk)
        buf.seek(0)

        print("Extracting G2PWModel ...")
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(parent_directory)

        if not final_dir.exists() and extract_dir.exists():
            os.rename(str(extract_dir), str(final_dir))

        print(f"G2PWModel ready at: {final_dir}")
    except Exception as e:
        print(f"Failed to ensure G2PWModel: {e}")


def ensure_data_from_hf() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Always ensure G2PWModel in src/text regardless of other dependencies
    ensure_g2pw_model()

    is_missing, items = need_download()
    if not is_missing:
        print("All required Data dependencies are present.")
        return

    print("Some Data dependencies are missing:")
    for scope, names in items:
        print(f"- {scope}: {', '.join(names)}")

    print(f"Downloading missing assets from Hugging Face repo: {REPO_ID} ...")
    local_dir = snapshot_download(repo_id=REPO_ID, local_dir=None, local_dir_use_symlinks=False)
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
            local_dir_use_symlinks=False
        )
        print(f"Chinese RoBERTa model downloaded to: {roberta_local_dir}")

    char_src_root = hf_root / "Data" / "character_model"
    if char_src_root.exists():
        CHAR_DIR.mkdir(parents=True, exist_ok=True)
        candidates = [p for p in char_src_root.iterdir() if p.is_dir()]
        candidates.sort(key=lambda p: p.name)
        if candidates:
            chosen = candidates[0]
            dst_char = CHAR_DIR / chosen.name
            if not dst_char.exists():
                dst_char.mkdir(parents=True, exist_ok=True)
            for path in chosen.rglob("*"):
                if path.is_dir():
                    (dst_char / path.relative_to(chosen)).mkdir(parents=True, exist_ok=True)
                else:
                    dst = dst_char / path.relative_to(chosen)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        dst.write_bytes(path.read_bytes())

    # Download audio resources
    audio_src_root = hf_root / "Data" / "audio_resources"
    if audio_src_root.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_candidates = [p for p in audio_src_root.iterdir() if p.is_dir()]
        audio_candidates.sort(key=lambda p: p.name)
        if audio_candidates:
            chosen_audio = audio_candidates[0]
            dst_audio_char = AUDIO_DIR / chosen_audio.name
            if not dst_audio_char.exists():
                dst_audio_char.mkdir(parents=True, exist_ok=True)
            for path in chosen_audio.rglob("*"):
                if path.is_dir():
                    (dst_audio_char / path.relative_to(chosen_audio)).mkdir(parents=True, exist_ok=True)
                else:
                    dst = dst_audio_char / path.relative_to(chosen_audio)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        dst.write_bytes(path.read_bytes())

    print("Data setup completed.")


if __name__ == "__main__":
    ensure_data_from_hf()


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "CHAR_DIR",
    "AUDIO_DIR",
    "REQUIRED_CN_HUBERT",
    "REQUIRED_OPENJTALK_DIR",
    "REQUIRED_CHINESE_ROBERTA_DIR",
    "CHAR_REQUIRED_FILES",
    "list_existing_characters",
    "list_existing_audio_characters",
    "character_missing_files",
    "need_download",
    "ensure_data_from_hf",
    "ensure_g2pw_model",
]


