import logging
from typing import List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class TextFrontend:
    def __init__(self):
        self._chinese_processor = None
        # English and Japanese are currently handled via module-level functions in their respective files,
        # but we import them lazily here.

    def _get_chinese_processor(self):
        if self._chinese_processor is None:
            logger.debug("Initializing Chinese G2P resources...")
            from ..Chinese.ChineseG2P import get_chinese_processor
            self._chinese_processor = get_chinese_processor()
        return self._chinese_processor

    def process_zh(self, text: str) -> Tuple[List[int], List[int], str]:
        """
        Process Chinese text.
        Returns: (phones_ids, word2ph, normalized_text)
        """
        # ChineseG2P.process returns: (normalized_text, phones, phones_ids, word2ph)
        normalized_text, _, phones_ids, word2ph = self._get_chinese_processor().process(text)
        return phones_ids, word2ph, normalized_text

    def process_en(self, text: str) -> List[int]:
        """
        Process English text.
        Returns: phones_ids
        """
        from ..English.EnglishG2P import english_to_phones
        return english_to_phones(text)

    def process_ja(self, text: str) -> List[int]:
        """
        Process Japanese text.
        Returns: phones_ids
        """
        from ..Japanese.JapaneseG2P import japanese_to_phones
        return japanese_to_phones(text)

    def warmup(self, language: str = "en") -> None:
        """
        Pre-load resources for the specified language to avoid first-call latency.
        """
        logger.debug(f"Warming up TextFrontend for language: {language}")
        if language == "zh":
            self._get_chinese_processor()
        elif language == "en":
            from ..English.EnglishG2P import get_g2p
            get_g2p()
        elif language == "ja":
            from ..Japanese.JapaneseG2P import get_ja_g2p
            try:
                get_ja_g2p()
            except ImportError:
                pass # Japanese G2P might not have a singleton getter yet

# Singleton management
_frontend_instance: Optional[TextFrontend] = None

def get_text_frontend() -> TextFrontend:
    global _frontend_instance
    if _frontend_instance is None:
        _frontend_instance = TextFrontend()
    return _frontend_instance

