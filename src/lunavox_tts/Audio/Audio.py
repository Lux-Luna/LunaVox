import os
import soundfile as sf
import soxr
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 音频时长建议范围 (秒)
MIN_DURATION_S = 3
MAX_DURATION_S = 10
# 在音频末尾追加的静音时长 (秒)
SILENCE_TO_APPEND_S = 0.3
# 模型期望的目标采样率
TARGET_SAMPLING_RATE = 16000


def load_audio(
        audio_path: str,
        target_sampling_rate: int = TARGET_SAMPLING_RATE
) -> Optional[np.ndarray]:
    try:
        wav, original_sr = sf.read(audio_path, dtype='float32')
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)  # 多声道转单声道。
        if original_sr != target_sampling_rate:
            wav = soxr.resample(wav, original_sr, target_sampling_rate, quality='hq')  # 重采样。

    except Exception as e:
        logger.error(f"Failed to load reference audio: {audio_path}. Error: {e}")
        return None

    # 检查音频长度是否在建议范围之外
    min_samples = int(MIN_DURATION_S * target_sampling_rate)
    max_samples = int(MAX_DURATION_S * target_sampling_rate)
    if not (min_samples <= wav.shape[0] <= max_samples):
        duration = len(wav) / target_sampling_rate
        logger.warning(
            f"The reference audio '{os.path.basename(audio_path)}' has a duration of {duration:.2f} seconds, "
            f"which is outside the recommended range of {MIN_DURATION_S} to {MAX_DURATION_S} seconds!"
        )

    # 创建并拼接静音
    silence_samples = int(SILENCE_TO_APPEND_S * target_sampling_rate)
    silence_array = np.zeros(silence_samples, dtype=np.float32)
    wav_processed = np.concatenate([wav, silence_array])

    # 为模型输入增加批次维度
    # wav_processed = np.expand_dims(wav_processed, axis=0)
    return wav_processed
