import time
import os
import json
import sys
from pathlib import Path

# Import LunaVox TTS from local src directory (support running from repo without installation)
SCRIPT_DIR = Path(__file__).parent
TUTORIAL_DIR = SCRIPT_DIR.parent  # Tutorial directory
REPO_ROOT = TUTORIAL_DIR.parent  # Go up two levels from v2_quick_tryout to repo root
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

# Import and run data_setup to ensure all required files are present
import data_setup
data_setup.ensure_data_from_hf()

import lunavox_tts as lunavox

# 设置环境变量使用Data目录下的本地文件
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')


def _resolve_reference_audio(language_folder: str):
    """
    Locate the first .wav file inside Data/audio_resources/<language_folder>.
    Returns the file path and an inferred transcript (filename stem).
    """
    audio_dir = REPO_ROOT / 'Data' / 'audio_resources' / language_folder
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Reference audio directory not found: {audio_dir}")
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    audio_file = wav_files[0]
    return str(audio_file), audio_file.stem

# 使用Data目录下的本地模型文件（使用 Data/character_model/v2/pretrained）
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2' / 'pretrained')
lunavox.load_character('pretrained', model_dir)

# 设置参考音频（自动查找 Data/audio_resources/Japanese 下的 .wav 文件）
audio_path, reference_text = _resolve_reference_audio('Japanese')
lunavox.set_reference_audio(
    'pretrained',
    audio_path,
    reference_text,
    audio_language='ja'
)

lunavox.tts(
    character_name='pretrained',
    text='こんにちは、ルナヴォックスです。日本語でお話しします。',
    play=True,  # Play the generated audio directly
    language='ja',  # 输出目标语言：日语
)

time.sleep(10)  # Add delay to ensure audio playback completes