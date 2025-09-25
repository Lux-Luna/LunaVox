import os

# (Optional) We recommend manually specifying the Hubert path for LunaVox.
# Download from Huggingface: https://huggingface.co/Lux-Luna/LunaVox
# Note: If this line is not set, LunaVox will automatically download the model from Huggingface.
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\chinese-hubert-base.onnx"

# (Optional) We recommend manually specifying the dictionary path for pyopenjtalk.
# Download from Huggingface: https://huggingface.co/Lux-Luna/LunaVox
# Note: If this line is not set, pyopenjtalk will automatically download the dictionary.
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\open_jtalk_dic_utf_8-1.11"

import lunavox_tts as lunavox

# Launch the LunaVox command-line client
lunavox.launch_command_line_client()
