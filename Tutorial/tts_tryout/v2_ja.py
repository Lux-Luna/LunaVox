import time
import os
import sys
import logging
from pathlib import Path

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ローカルの src を sys.path に追加
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from lunavox_tts.Utils.EnvManager import env_manager

# --- オプション: 環境設定 ---
# 特定の実行モードを強制する場合は、以下の行のコメントを解除してください。デフォルトは "cpu" です。
# env_manager.set_mode("cpu")
# env_manager.set_mode("gpu")

if not env_manager.ensure_environment():
    print(f"\n環境が {env_manager.get_mode().upper()} に更新されました。このスクリプトを再実行してください。")
    sys.exit(0)

import lunavox_tts as lunavox

# ローカル環境設定
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'CharacterData' / 'audio' / language
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem

# 1. Persona のロード (推奨モード)
# 事前に抽出・固化された特徴量 (luna_ja) を使用し、実行時の音声解析をスキップします。
# このモードは、起動が速く、メモリ使用量が少なく、音質が安定しています。
char_name = 'luna_ja'
persona_dir = str(REPO_ROOT / 'CharacterData' / 'character' / 'luna_ja')
model_dir = str(REPO_ROOT / 'CharacterData' / 'model' / 'v2' / 'pretrained')

lunavox.load_persona(char_name, persona_dir)
lunavox.load_character(char_name, model_dir)

# 2. 代替案: 参照オーディオモード (コメントアウト)
# 特定の WAV ファイルを使用してリアルタイムでボイスクローニングを行う場合に使用します。
# 注意: 起動するたびに特徴量の再抽出が行われます。
"""
audio_path, reference_text = resolve_reference('Japanese')
lunavox.set_reference_audio(char_name, audio_path, reference_text, audio_language='ja')
"""

# 3. テキスト読み上げ (TTS)
lunavox.tts(
    character_name=char_name,
    text='こんにちは、ルナヴォックスです。',
    play=True,
    language='ja'
)

# 再生完了まで待機
time.sleep(5)