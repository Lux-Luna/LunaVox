import time
import os
import json
import sys
from pathlib import Path

# Import LunaVox TTS from local src directory (support running from repo without installation)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent  # Go up one level from Tutorial to repo root
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

# Import and run data_setup to ensure all required files are present
import data_setup
data_setup.ensure_data_from_hf()

import lunavox_tts as lunavox

# 设置环境变量使用Data目录下的本地文件
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')

# 使用Data目录下的本地模型文件（使用 Data/character_model/v2/pretrained）
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2' / 'pretrained')
lunavox.load_character('pretrained', model_dir)

# 设置参考音频（使用 Data/audio_resources/pretrained）
audio_path = str(
    REPO_ROOT
    / 'Data'
    / 'audio_resources'
    / 'pretrained'
    / '私は天使なんかじゃないわ。病院なんてないわよ。誰も病まないから。みんな死んでるから。.wav'
)
lunavox.set_reference_audio(
    'pretrained',
    audio_path,
    "私は天使なんかじゃないわ。病院なんてないわよ。誰も病まないから。みんな死んでるから。",
    audio_language='ja'
)

lunavox.tts(
    character_name='pretrained',
    text='こんにちは、ルナヴォックスです。日本語でお話しします。',
    play=True,  # Play the generated audio directly
    language='ja',  # 输出目标语言：日语
)

time.sleep(10)  # Add delay to ensure audio playback completes