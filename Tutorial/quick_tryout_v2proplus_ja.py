"""
Quick tryout script for yuzuki_yukari v2ProPlus model - Japanese synthesis.

This script demonstrates v2ProPlus inference with the converted yuzuki_yukari model.
"""
import time
import os
import sys
from pathlib import Path

# Import LunaVox TTS from local src directory
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import lunavox_tts as lunavox
import data_setup
data_setup.ensure_data_from_hf()
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')

model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2_pro_plus' / 'pretrained')
lunavox.load_character('pretrained', model_dir)

# Check model version
# No extra version/output checks to keep parity with quick_tryout_ja.py

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
    language='ja',
)

time.sleep(10)  # Ensure audio playback completes
