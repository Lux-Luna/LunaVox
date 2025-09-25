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

# 使用Data目录下的本地模型文件
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'yuzuki_yukari')
lunavox.load_character('yuzuki_yukari', model_dir)

# 设置参考音频（参考语言为日语）
audio_path = str(REPO_ROOT / 'Data' / 'audio_resources' / 'yuzuki_yukari' / "ありがとうございます。おひさしぶりです。.wav")
lunavox.set_reference_audio('yuzuki_yukari', audio_path, "ありがとうございます。おひさしぶりです。", audio_language='ja')

lunavox.tts(
    character_name='yuzuki_yukari',
    text='こんにちは、ルナヴォックスです。日本語でお話しします。',
    play=True,  # Play the generated audio directly
    language='ja',  # 输出目标语言：日语
)

time.sleep(10)  # Add delay to ensure audio playback completes