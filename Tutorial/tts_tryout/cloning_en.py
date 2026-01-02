import time
from pathlib import Path
import common_setup; common_setup.configure_paths()
import lunavox_tts as lunavox

# Setup paths helper
REPO_ROOT = Path(__file__).parent.parent.parent

def get_reference_audio(language: str):
    """Finds the first .wav file in the standard audio directory."""
    audio_dir = REPO_ROOT / 'lunavoxData' / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    return wav_files[0]

# 1. Initialize (Loads 'luna_en' Model)
# We load the standard character. Using the facade ensures models are ready.
lunavox.initialize_tts('luna_en')

# 2. Configure Voice Cloning (Reference Audio)
# This overrides the default Persona with a specific audio file.
ref_wav = get_reference_audio('English')
print(f"Cloning voice from: {ref_wav.name}")

lunavox.set_reference_audio(
    character_name='luna_en',
    audio_path=str(ref_wav),
    # Assuming the filename represents the text content for this demo
    audio_text=ref_wav.stem, 
    audio_language='en'
)

# 3. Speak
print("Generating cloned audio...")
lunavox.tts(
    character_name='luna_en',
    text='I am now running in Reference Audio mode, cloning the voice you selected.',
    play=True,
    language='en'
)

time.sleep(5)
