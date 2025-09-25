import re
import logging

logger = logging.getLogger(__name__)

MIN_SENTENCE_LENGTH = 5

# 定义用于分割句子的标点
SENTENCE_TERMINATORS = "、。！？…"
# 定义有效字符的正则表达式，用于精确计算长度
VALID_CHAR_PATTERN = re.compile(
    r'[\u3040-\u309F'  # 平假名
    r'\u30A0-\u30FF'  # 片假名
    r'\u4E00-\u9FFF'  # 汉字
    r'a-zA-Z'  # 半角字母
    r'\uFF21-\uFF3A\uFF41-\uFF5A'  # 全角字母
    r'0-9'  # 半角数字
    r'\uFF10-\uFF19'  # 全角数字
    r']'
)


def get_valid_text_length(sentence: str) -> int:
    return len(VALID_CHAR_PATTERN.findall(sentence))


def split_japanese_text(long_text: str) -> list[str]:
    if not long_text:
        return []
    # 使用正向后行断言 `(?<=...)` 来在分割时保留分隔符
    raw_sentences = re.split(f'(?<=[{SENTENCE_TERMINATORS}])', long_text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        return [long_text] if long_text.strip() else []

    final_sentences = []
    for sentence in raw_sentences:
        clean_len = get_valid_text_length(sentence)
        # 如果final_sentences不为空，且上一句的有效长度也小于最小长度，或者当前句本身就小于最小长度，则合并
        if final_sentences and clean_len < MIN_SENTENCE_LENGTH:
            final_sentences[-1] += sentence
        else:
            final_sentences.append(sentence)
    return final_sentences
