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
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem

# 1. 加载 Persona 音色固化 (v2ProPlus 推荐模式)
# 使用固化后的 Persona (luna_zh)，其中包含预计算的声纹向量 (Speaker Vector)。
# 这样可以跳过运行时的 HuBERT 模型提取和声纹分析，大幅节省启动时间和内存。
char_name = 'luna_v2pp_zh'
persona_dir = str(REPO_ROOT / 'CharacterData' / 'character' / 'luna_zh')
model_dir = str(REPO_ROOT / 'CharacterData' / 'model' / 'v2_pro_plus' / 'pretrained')

lunavox.load_persona(char_name, persona_dir)
lunavox.load_character(char_name, model_dir)

# 2. 备选方案：参考音频模式 (已注释)
# 如果您想要使用特定的 WAV 文件进行实时声音克隆，请使用此代码。
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
