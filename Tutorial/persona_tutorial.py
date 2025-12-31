import os
import sys
# 1. Setup paths
# Move up one level to find the project root
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import lunavox_tts as lunavox

# Paths for resources
MODEL_PATH = os.path.join(PROJ_ROOT, "CharacterData", "model", "v2", "pretrained")
REF_WAV = os.path.join(PROJ_ROOT, "CharacterData", "audio", "English", 
                       "First get into position like this, then move like that. Yep, thats it..wav")
REF_TEXT = "First get into position like this, then move like that. Yep, that's it."
PERSONA_SAVE_DIR = "./cached_persona"

def persona_tutorial():
    # --- STEP 1: Load Model ---
    # You only need to do this once for the character
    print("Loading model...")
    lunavox.load_character("my_character", MODEL_PATH)

    # --- STEP 2: Create Persona (One-time) ---
    # This extracts features and saves them to a folder.
    # You won't need the original WAV file after this!
    print("Creating persona...")
    lunavox.create_persona(
        character_name="my_character",
        audio_path=REF_WAV,
        audio_text=REF_TEXT,
        save_dir=PERSONA_SAVE_DIR,
        audio_language="en"
    )

    # --- STEP 3: Load Persona ---
    # Load the character's digital "soul" without re-processing audio.
    # Extremely fast and lightweight!
    print("Loading persona...")
    lunavox.load_persona("my_character", PERSONA_SAVE_DIR)

    # --- STEP 4: Reference-Free TTS ---
    # Enjoy seamless voice generation without providing WAV paths anymore.
    print("Generating speech (Reference-Free)...")
    lunavox.tts(
        character_name="my_character",
        text="This is magic! I don't need my original audio anymore.",
        save_path="tutorial_output.wav",
        language="en"
    )
    print("Done! Check 'tutorial_output.wav'")

if __name__ == "__main__":
    persona_tutorial()
