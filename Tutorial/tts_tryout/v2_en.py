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

# --- Option: Environment Setup ---
# Uncomment the following lines to force a specific execution mode. Default is "cpu".
# env_manager.set_mode("cpu")
# env_manager.set_mode("gpu")

if not env_manager.ensure_environment():
    print(f"\nEnvironment updated to {env_manager.get_mode().upper()}. Please RE-RUN this script.")
    sys.exit(0)

import lunavox_tts as lunavox

# Local environment configuration
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem

# 1. Load Persona (Recommended Mode)
# This uses pre-solidified features (luna_en) and does not require a reference audio file at runtime.
# This mode offers higher quality, faster startup, and lower GPU VRAM overhead.
char_name = 'luna_en'
persona_dir = str(REPO_ROOT / 'CharacterData' / 'character' / 'luna_en')
model_dir = str(REPO_ROOT / 'CharacterData' / 'model' / 'v2' / 'pretrained')

lunavox.load_persona(char_name, persona_dir)
lunavox.load_character(char_name, model_dir)

# 2. Alternative: Reference Audio Mode (Commented out)
# Use this if you want to perform real-time voice cloning using a specific WAV file.
# Note: This will re-extract features every time the service restarts.
"""
audio_path, reference_text = resolve_reference('English')
lunavox.set_reference_audio(char_name, audio_path, reference_text, audio_language='en')
"""

# 3. Text-to-Speech (TTS)
lunavox.tts(
    character_name=char_name,
    text='Hi, This is LunaVox speaking English.',
    play=True,
    language='en'
)

# Wait for playback to finish
time.sleep(5)
