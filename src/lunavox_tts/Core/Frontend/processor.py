"""
TextProcessor - Centralized text processing and splitting.
"""
import re
import logging
from typing import List, Dict
from .registry import list_supported_languages

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles sentence splitting and language detection/splitting."""
    
    @staticmethod
    def split_sentences(text: str, language: str) -> List[str]:
        """
        Split text into sentences based on language-specific rules.
        """
        text = text.strip()
        if not text:
            return []
            
        lang = language.lower()
        if lang == "ja":
            from ...Languages.Japanese.Split import split_japanese_text
            return split_japanese_text(text)
        elif lang == "en":
            # Better regex for English sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        elif lang == "zh":
            # Split by Chinese punctuation
            sentences = re.split(r'([。！？…])', text)
            result = []
            for i in range(0, len(sentences) - 1, 2):
                result.append(sentences[i] + sentences[i+1])
            if len(sentences) % 2 == 1 and sentences[-1]:
                if not sentences[-1].strip(): pass
                else: result.append(sentences[-1])
            return [s.strip() for s in result if s.strip()]
        else:
            # Fallback/Generic
            sentences = re.split(r'(?<=[.!?。！？])\s*', text)
            return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def split_language(text: str) -> List[Dict[str, str]]:
        """
        Split a mixed-language string into components with language labels.
        Currently handles Zh/En mixing.
        """
        pattern_eng = re.compile(r"[a-zA-Z]+")
        split = re.split(pattern_eng, text)
        matches = pattern_eng.findall(text)

        result = []
        for i, part in enumerate(split):
            if part.strip():
                result.append({'language': 'zh', 'content': part})
            if i < len(matches):
                result.append({'language': 'en', 'content': matches[i]})

        return result

    @staticmethod
    def normalize_punctuation(text: str, language: str) -> str:
        """Add recommended leading/trailing punctuation for better TTS stability."""
        if language == "ja" or language == "zh":
             if not text.startswith("。"): text = "。" + text
             if not text.endswith(("。", "！", "？", "…")): text = text + "。"
        else:
             if not text.startswith("."): text = "." + text
             if not text.endswith((".", "!", "?")): text = text + "."
        return text

text_processor = TextProcessor()
