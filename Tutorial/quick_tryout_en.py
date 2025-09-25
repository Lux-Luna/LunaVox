import time
import os
import sys
from pathlib import Path

# Import LunaVox TTS from local src directory (support running from repo without installation)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent  # Go up one level from Tutorial to repo root
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import lunavox_tts as lunavox

# 使用Data目录下的本地文件
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')

# 加载模型
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'yuzuki_yukari')
lunavox.load_character('yuzuki_yukari', model_dir)

# 设置参考音频（参考语言为日语）
audio_path = str(REPO_ROOT / 'Data' / 'audio_resources' / 'yuzuki_yukari' / "ありがとうございます。おひさしぶりです。.wav")
lunavox.set_reference_audio('yuzuki_yukari', audio_path, "ありがとうございます。おひさしぶりです。", audio_language='ja')

# 合成英文
lunavox.tts(
    character_name='yuzuki_yukari',
    text='Hello, this is LunaVox speaking English.',
    play=True,
    language='en',  # 输出目标语言：英语
)

time.sleep(10)


