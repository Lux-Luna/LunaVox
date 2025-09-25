from ..Utils.Utils import LRUCacheDict
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..English.EnglishG2P import english_to_phones
from ..Utils.Constants import BERT_FEATURE_DIM
from ..Audio.Audio import load_audio
from ..ModelManager import model_manager

import os
import numpy as np
import soxr
from typing import Optional


class ReferenceAudio:
    _prompt_cache: dict[str, 'ReferenceAudio'] = LRUCacheDict(
        capacity=int(os.getenv('Max_Cached_Reference_Audio', '10')))

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str = 'auto'):
        if prompt_wav in cls._prompt_cache:
            instance = cls._prompt_cache[prompt_wav]
            if instance.text != prompt_text:  # 如果文本与缓存内记录的不同，则更新。
                instance.set_text(prompt_text)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[prompt_wav] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str = 'auto'):
        if hasattr(self, '_initialized'):
            return

        # 文本相关。
        self.text: str = prompt_text
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.set_text(prompt_text, language)

        # 音频相关。
        self.audio_32k: Optional[np.ndarray] = load_audio(
            audio_path=prompt_wav,
            target_sampling_rate=32000
        )
        audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality='hq')
        audio_16k = np.expand_dims(audio_16k, axis=0)  # 增加 Batch_Size 维度

        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
            None, {'input_values': audio_16k}
        )[0]

        self._initialized = True

    def set_text(self, prompt_text: str, language: str = 'auto') -> None:
        self.text = prompt_text
        # Choose G2P based on language hint or heuristic.
        if language == 'en' or (language == 'auto' and _looks_english(prompt_text)):
            ids = english_to_phones(prompt_text)
        else:
            ids = japanese_to_phones(prompt_text)
        self.phonemes_seq = np.array([ids], dtype=np.int64)
        self.text_bert: Optional[np.ndarray] = np.zeros((self.phonemes_seq.shape[1], BERT_FEATURE_DIM),
                                                        dtype=np.float32)

    @classmethod
    def clear_cache(cls) -> None:
        """清空 ReferenceAudio 的缓存"""
        cls._prompt_cache.clear()


def _looks_english(text: str) -> bool:
    # Rough heuristic: contains Latin letters significantly and few kana/kanji
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    non_ascii = sum(not ch.isascii() and not ch.isspace() for ch in text)
    return ascii_letters > 0 and ascii_letters >= non_ascii
