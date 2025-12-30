import time
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local src to sys.path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from lunavox_tts.Utils.EnvManager import env_manager

# --- OPTIONAL: Environment Configuration ---
# Uncomment to force a specific runtime mode. Default is "cpu".
env_manager.set_mode("cpu")
# env_manager.set_mode("gpu")

if not env_manager.ensure_environment():
    print(f"\nEnvironment updated to {env_manager.get_mode().upper()}. Please RE-RUN this script.")
    sys.exit(0)

import lunavox_tts as lunavox

# Local environment settings
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio_resources' / language
    wav_file = next(audio_dir.glob("*.wav"))
    return str(wav_file), wav_file.stem

# 1. Load v2 Pro Plus Model
model_dir = str(REPO_ROOT / 'CharacterData' / 'character_model' / 'v2_pro_plus' / 'pretrained')
lunavox.load_character('pretrained_v2pp', model_dir)

# 2. Set Reference Audio
audio_path, reference_text = resolve_reference('English')
lunavox.set_reference_audio('pretrained_v2pp', audio_path, reference_text, audio_language='en')

# 3. Text-to-Speech
lunavox.tts(
    character_name='pretrained_v2pp',
    text='Hi, This is the LunaVox speaking English.',
    play=True,
    language='en'
)

# Keep process alive for playback
time.sleep(5)
