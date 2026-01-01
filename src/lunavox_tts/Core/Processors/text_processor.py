"""
Text Processor - Preprocessing logic extracted from LunaVoxEngine.

Handles text normalization and punctuation padding for TTS.
"""
import re
from typing import Optional


# Punctuation that can serve as sentence boundaries
SENTENCE_ENDINGS = (".", "。", "?", "？", "!", "！", "…", "—", "-")


def preprocess_text(text: str, language: str = "ja") -> str:
    """
    Apply preprocessing to input text before TTS.
    
    This includes:
    - Adding leading punctuation to prevent first-sentence truncation
    - Adding trailing punctuation to prevent last-sentence truncation
    
    Args:
        text: Raw input text.
        language: Language code ('ja', 'en', 'zh').
        
    Returns:
        Preprocessed text ready for TTS.
    """
    if not text or not text.strip():
        return text
    
    # Add leading punctuation if not present
    # This prevents the first sentence from being cut off
    if not text.startswith("。") and not text.startswith("."):
        text = "。" + text
    
    # Add trailing punctuation if not present
    # This prevents the last sentence from being truncated
    if not text.strip().endswith(SENTENCE_ENDINGS):
        text = text + "。"
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_by_language(text: str) -> list:
    """
    Split text into segments by language (Chinese vs English).
    
    Returns list of dicts with 'language' and 'content' keys.
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
