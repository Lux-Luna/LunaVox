import os

# (Optional) We recommend manually specifying the Hubert path for LunaVox.
# Download from Huggingface: https://huggingface.co/Lux-Luna/LunaVox
# Note: If this line is not set, LunaVox will automatically download the model from Huggingface.
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\chinese-hubert-base.onnx"

# (Optional) We recommend manually specifying the dictionary path for pyopenjtalk.
# Download from Huggingface: https://huggingface.co/Lux-Luna/LunaVox
# Note: If this line is not set, pyopenjtalk will automatically download the dictionary.
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\open_jtalk_dic_utf_8-1.11"

# (Optional) You can set the number of cached character models and reference audios.
os.environ['Max_Cached_Character_Models'] = '3'
os.environ['Max_Cached_Reference_Audio'] = '10'

# Make sure to set environment variables before importing LunaVox.
import lunavox_tts as lunavox
import time

lunavox.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Replace with the folder containing the ONNX model
)

lunavox.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Use the same character name as above
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Replace with path to your reference audio file
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Replace with the text corresponding to the reference audio
)

lunavox.tts(
    character_name='<CHARACTER_NAME>',  # Use the same character name
    text="<TEXT_TO_SYNTHESIZE>",  # Replace with the text you want to synthesize
    play=True,  # Whether to play the audio immediately
    split_sentence=True,  # Whether to split sentences before TTS
    save_path="<OUTPUT_AUDIO_PATH>",  # Replace with path to save the audio file
)

time.sleep(10)  # Since play=True, wait for playback to finish
