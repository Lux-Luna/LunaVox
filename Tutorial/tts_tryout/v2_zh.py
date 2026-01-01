import time
import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 将本地 src 添加到 sys.path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from lunavox_tts.Utils.EnvManager import env_manager

# --- 可选：环境配置 ---
# 取消注释以强制指定运行模式。默认为 "cpu"。
# env_manager.set_mode("cpu")
# env_manager.set_mode("gpu")

if not env_manager.ensure_environment():
    print(f"\n环境已更新为 {env_manager.get_mode().upper()}。请重新运行此脚本。")
    sys.exit(0)

import lunavox_tts as lunavox

# 本地环境设置
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'lunavoxData' / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'lunavoxData' / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem

# 1. 加载 Persona 音色固化 (推荐模式)
# 使用预先提取并固化的特征 (luna_zh)，无需在运行时提供参考音频。
# 这种模式启动更快，显存占用更低，且音质更加稳定。
char_name = 'luna_zh'
persona_dir = str(REPO_ROOT / 'lunavoxData' / 'CharacterData' / 'character' / 'luna_zh')
model_dir = str(REPO_ROOT / 'lunavoxData' / 'CharacterData' / 'model' / 'v2' / 'pretrained')

lunavox.load_persona(char_name, persona_dir)
lunavox.load_character(char_name, model_dir)

# 2. 备选方案：参考音频模式 (已注释)
# 如果您想要使用特定的 WAV 文件进行实时声音克隆，请使用此代码。
# 注意：每次重启服务都会重新提取特征。
"""
audio_path, reference_text = resolve_reference('Chinese')
lunavox.set_reference_audio(char_name, audio_path, reference_text, audio_language='zh')
"""

# 3. 文本转语音 (TTS)
lunavox.tts(
    character_name=char_name,
    text='你好，这是一次中文语音合成测试。',
    play=True,
    language='zh'
)

# 等待播放完成
time.sleep(5)
