import time
import os
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import lunavox_tts as lunavox

# Use Data directory
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')

# Load model
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2' / 'pretrained_fp16')
lunavox.load_character('pretrained', model_dir)

# Set reference audio (Chinese)
audio_path = str(REPO_ROOT / 'Data' / 'audio_resources' / 'Chinese' / '所以命运才是终极的知识，对吗？.wav')
reference_text = "所以命运才是终极的知识，对吗？"
lunavox.set_reference_audio(
    'pretrained',
    audio_path,
    reference_text,
    audio_language='zh'
)

# Test Chinese TTS (Triggers BERT)
print("Testing Chinese TTS...")
lunavox.tts(
    character_name='pretrained',
    text='你好，这是正在测试中文语音合成。',
    play=False,
    language='zh',
)

