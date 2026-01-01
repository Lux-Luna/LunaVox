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
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem

# 1. Load Universal Persona (Recommended for v2ProPlus)
# Using Universal Persona (luna_en) which includes pre-computed global embeddings.
# This skips HuBERT, Speaker Vector extraction, AND prompt_encoder loading at runtime.
# The load_persona() call automatically loads the base v2ProPlus models.
char_name = 'luna_v2pp_en'
persona_dir = str(REPO_ROOT / 'CharacterData' / 'character' / 'luna_en')

# Note: load_persona() will auto-load v2ProPlus models AND skip prompt_encoder
# since the persona has cached global embeddings. No need to call load_character separately.
lunavox.load_persona(char_name, persona_dir)

# 2. Alternative: Reference Audio Mode (Commented out)
# Use this if you want to clone a voice from a specific WAV file in real-time.
# In this mode, prompt_encoder WILL be loaded to compute global embeddings on-the-fly.
"""
model_dir = str(REPO_ROOT / 'CharacterData' / 'model' / 'v2_pro_plus' / 'pretrained')
lunavox.load_character(char_name, model_dir)  # This loads prompt_encoder
audio_path, reference_text = resolve_reference('English')
lunavox.set_reference_audio(char_name, audio_path, reference_text, audio_language='en')
"""

# 3. Text-to-Speech
lunavox.tts(
    character_name=char_name,
    text='Hi, This is LunaVox speaking English.',
    play=True,
    language='en'
)

# Keep process alive for playback
time.sleep(5)
