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

# 确保数据依赖项存在
import data_setup
data_setup.ensure_data_from_hf()

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
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'Data' / 'audio_resources' / language
    wav_file = next(audio_dir.glob("*.wav"))
    return str(wav_file), wav_file.stem

# 1. 加载角色模型
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2' / 'pretrained')
lunavox.load_character('pretrained', model_dir)

# 2. 设置参考音频 (使用 Chinese 文件夹下的第一个 .wav 文件)
audio_path, reference_text = resolve_reference('Chinese')
lunavox.set_reference_audio('pretrained', audio_path, reference_text, audio_language='zh')

# 3. 文本转语音 (TTS)
lunavox.tts(
    character_name='pretrained',
    text='你好，我是 LunaVox。现在正在为您演示中文语音合成。',
    play=True,
    language='zh'
)

# 等待播放完成
time.sleep(5)
