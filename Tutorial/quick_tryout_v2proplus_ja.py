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

print("="*60)
print("yuzuki_yukari v2ProPlus - Japanese Quick Tryout")
print("="*60)
print()

# Setup environment paths
print("[1/4] Setting up environment...")
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base.onnx')
os.environ['OPEN_JTALK_DICT_DIR'] = str(REPO_ROOT / 'Data' / 'open_jtalk_dic_utf_8-1.11')

# Check if HuBERT model exists
hubert_path = Path(os.environ['HUBERT_MODEL_PATH'])
if not hubert_path.exists():
    print(f"  [ERROR] HuBERT model not found at: {hubert_path}")
    print("  Please ensure chinese-hubert-base.onnx is in LunaVox/Data/")
    sys.exit(1)

print(f"  [OK] HuBERT model: {hubert_path}")
print(f"  [OK] Open JTalk dict: {os.environ['OPEN_JTALK_DICT_DIR']}")
print()

# Load converted v2ProPlus model
print("[2/4] Loading yuzuki_yukari v2ProPlus model...")
# Use the converted model from parent directory (with dynamic length support!)
model_dir = str(Path(__file__).parent.parent.parent / 'test_output' / 'yuzuki_yukari_v2proplus')

if not Path(model_dir).exists():
    print(f"  [ERROR] Model not found at: {model_dir}")
    print("  Please run model conversion first:")
    print("    python -c \"import sys; sys.path.insert(0, 'LunaVox/src'); import lunavox_tts as lunavox; lunavox.convert_to_onnx('v2proplus_resource/yuzuki_yukari/yuzuki_yukari.ckpt', 'v2proplus_resource/yuzuki_yukari/yuzuki_yukari.pth', 'test_output/yuzuki_yukari_v2proplus_onnx')\"")
    sys.exit(1)

lunavox.load_character('yuzuki_yukari', model_dir)
print(f"  [OK] Model loaded from: {model_dir}")

# Check model version
from lunavox_tts.ModelManager import model_manager
version = model_manager.get_character_version('yuzuki_yukari')
print(f"  [OK] Model version: {version}")
print()

# Set reference audio (use the same reference as quick_tryout_ja.py)
print("[3/4] Setting reference audio...")
audio_resources_dir = REPO_ROOT / 'Data' / 'audio_resources' / 'yuzuki_yukari'

# Use the first reference audio
reference_file = "ありがとうございます。おひさしぶりです。.wav"
audio_path = str(audio_resources_dir / reference_file)

if not Path(audio_path).exists():
    print(f"  [ERROR] Reference audio not found: {audio_path}")
    print("  Please ensure audio resources are in LunaVox/Data/audio_resources/yuzuki_yukari/")
    sys.exit(1)

reference_text = "ありがとうございます。おひさしぶりです。"

lunavox.set_reference_audio(
    'yuzuki_yukari',
    audio_path,
    reference_text,
    audio_language='ja'
)

print(f"  [OK] Reference: {reference_file}")
print(f"  [OK] Text: {reference_text}")
print()

# Generate Japanese speech
print("[4/4] Generating Japanese speech...")
test_text = '我可以按你的模型给一版最小复现与改造补丁。'
output_path = str(Path(__file__).parent.parent.parent / 'test_output' / 'v2proplus_ja_output.wav')

# Create output directory
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"  Text: {test_text}")
print(f"  Output: {output_path}")
print()
print("  Synthesizing... (this may take a few seconds)")

lunavox.tts(
    character_name='yuzuki_yukari',
    text=test_text,
    play=False,  # Don't play, just save
    split_sentence=True,  # Split with reduced pause duration
    language='zh',
    save_path=output_path
)

print()

# Check if output was created
if Path(output_path).exists():
    file_size = Path(output_path).stat().st_size
    print("="*60)
    print("SUCCESS!")
    print("="*60)
    print()
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print()
    print("  Play this file to verify the audio quality.")
    print("  If you hear clear Japanese speech, the v2ProPlus")
    print("  implementation is working correctly!")
    print()
else:
    print("="*60)
    print("FAILED")
    print("="*60)
    print()
    print(f"  Output file was not created: {output_path}")
    print("  Please check the logs above for errors.")
    print()

print("="*60)
