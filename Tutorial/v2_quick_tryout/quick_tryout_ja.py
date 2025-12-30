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

# データ依存関係の確認
import data_setup
data_setup.ensure_data_from_hf()

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
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'Data' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

def resolve_reference(language: str):
    audio_dir = REPO_ROOT / 'Data' / 'audio_resources' / language
    wav_file = next(audio_dir.glob("*.wav"))
    return str(wav_file), wav_file.stem

# 1. キャラクタモデルのロード
model_dir = str(REPO_ROOT / 'Data' / 'character_model' / 'v2' / 'pretrained')
lunavox.load_character('pretrained', model_dir)

# 2. 参照オーディオの設定 (Japanese フォルダ内の最初の .wav ファイル)
audio_path, reference_text = resolve_reference('Japanese')
lunavox.set_reference_audio('pretrained', audio_path, reference_text, audio_language='ja')

# 3. テキスト読み上げ (TTS)
lunavox.tts(
    character_name='pretrained',
    text='こんにちは、ルナヴォックスです。日本語の音声合成テストを開始します。',
    play=True,
    language='ja'
)

# 再生完了まで待機
time.sleep(5)