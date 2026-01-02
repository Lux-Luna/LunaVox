import time
import common_setup; common_setup.configure_paths()
import lunavox_tts as lunavox

# 1. Initialize (Loads 'luna_ja' Persona + v2/v2pp Model automatically)
# To use v2_pro_plus model, change version='v2' to 'v2_pro_plus'
# To force device, add argument: device='cpu' or device='gpu'
lunavox.initialize_tts('luna_ja', version='v2')

# 2. Speak
print("Generating audio...")
lunavox.tts(
    character_name='luna_ja',
    text='こんにちは、ルナヴォックスです。',
    play=True,
    language='ja'
)

# Wait for playback
time.sleep(5)
