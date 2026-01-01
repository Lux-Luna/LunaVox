"""
Text Processing - Preprocessing logic for TTS.

Handles punctuation padding and text normalization.
"""
import re

SENTENCE_ENDINGS = (".", "。", "?", "？", "!", "！", "…", "—", "-")


def preprocess_text(text: str, language: str = "ja") -> str:
    """Apply preprocessing to input text before TTS."""
    if not text or not text.strip():
        return text
    
    if not text.startswith("。") and not text.startswith("."):
        text = "。" + text
    
    if not text.strip().endswith(SENTENCE_ENDINGS):
        text = text + "。"
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_by_language(text: str) -> list:
    """Split text into segments by language (Chinese vs English)."""
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
