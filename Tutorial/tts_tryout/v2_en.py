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
# env_manager.set_mode("cpu")
# env_manager.set_mode("gpu")

if not env_manager.ensure_environment():
    print(f"\nEnvironment updated to {env_manager.get_mode().upper()}. Please RE-RUN this script.")
    sys.exit(0)

import lunavox_tts as lunavox

# Local environment settings
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio' / language
    wav_file = next(audio_dir.glob("*.wav"))
    return str(wav_file), wav_file.stem

# 1. Load Persona (Default: luna_en)
character_name = 'luna_en'
persona_dir = str(REPO_ROOT / 'CharacterData' / 'character' / 'luna_en')
lunavox.load_persona(character_name, persona_dir)

# --- Option: Reference Audio Mode (Commented out) ---
# To use reference audio directly instead of a persona, uncomment the lines below 
# and comment out the "Load Persona" section above.
# 
# model_dir = str(REPO_ROOT / 'CharacterData' / 'model' / 'v2' / 'pretrained')
# audio_path, reference_text = resolve_reference('English')
# lunavox.load_character(character_name, model_dir)
# lunavox.set_reference_audio(character_name, audio_path, reference_text, audio_language='en')

# 2. Text-to-Speech (TTS)
lunavox.tts(
    character_name=character_name,
    text='Hi, this is LunaVox speaking English.',
    play=True,
    language='en'
)

# Wait for playback to complete
time.sleep(5)
