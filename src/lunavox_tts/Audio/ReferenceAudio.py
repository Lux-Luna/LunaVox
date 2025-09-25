import os
from typing import Optional

import numpy as np
import soxr

from ..Audio.Audio import load_audio
from ..Chinese.ChineseG2P import chinese_clean_g2p_and_norm
from ..Chinese.ZhBert import compute_bert_phone_features
from ..English.EnglishG2P import english_to_phones
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..ModelManager import model_manager
from ..Utils.Constants import BERT_FEATURE_DIM
from ..Utils.Shared import context
from ..Utils.Utils import LRUCacheDict


class ReferenceAudio:
    _prompt_cache: dict[tuple[str, str], "ReferenceAudio"] = LRUCacheDict(
        capacity=int(os.getenv("Max_Cached_Reference_Audio", "10"))
    )

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str = "auto"):
        key = (prompt_wav, (language or "auto"))
        if key in cls._prompt_cache:
            instance = cls._prompt_cache[key]
            if instance.text != prompt_text or instance.language != key[1]:
                instance.set_text(prompt_text, language)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[key] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str = "auto"):
        if hasattr(self, "_initialized"):
            return

        self.text: str = prompt_text
        self.language: str = language or "auto"
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.set_text(prompt_text, language)

        self.audio_32k: Optional[np.ndarray] = load_audio(
            audio_path=prompt_wav,
            target_sampling_rate=32000,
        )
        audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality="hq")
        audio_16k = np.expand_dims(audio_16k, axis=0)

        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
            None, {"input_values": audio_16k}
        )[0]

        self._initialized = True

    def set_text(self, prompt_text: str, language: str = "auto") -> None:
        self.text = prompt_text
        lang = _decide_language(prompt_text, language)
        self.language = lang

        if lang == "en":
            ids = english_to_phones(prompt_text)
            word2ph: list[int] = []
            norm_text = ""
        elif lang == "zh":
            ids, word2ph, norm_text = chinese_clean_g2p_and_norm(prompt_text)
        else:
            ids = japanese_to_phones(prompt_text)
            word2ph = []
            norm_text = ""

        self.phonemes_seq = np.array([ids], dtype=np.int64)
        bert_matrix = _compute_reference_bert(lang, norm_text, word2ph, len(ids))
        self.text_bert = bert_matrix

        if lang in {"ja", "en", "zh"}:
            context.current_language = lang

    @classmethod
    def clear_cache(cls) -> None:
        cls._prompt_cache.clear()


def _decide_language(text: str, language: Optional[str]) -> str:
    lang = (language or "auto").lower()
    if lang == "auto":
        if _looks_english(text):
            return "en"
        if _looks_chinese(text):
            return "zh"
        return "ja"
    if lang in {"ja", "en", "zh"}:
        return lang
    return "ja"


def _compute_reference_bert(
    language: str, norm_text: str, word2ph: list[int], phone_len: int
) -> np.ndarray:
    if language == "zh" and phone_len:
        bert = compute_bert_phone_features(norm_text, word2ph)
        if bert.shape[0] == phone_len:
            return bert.astype(np.float32)
    return np.zeros((phone_len, BERT_FEATURE_DIM), dtype=np.float32)


def _looks_english(text: str) -> bool:
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    non_ascii = sum(not ch.isascii() and not ch.isspace() for ch in text)
    return ascii_letters > 0 and ascii_letters >= non_ascii


def _looks_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


