import re
from typing import List

from g2p_en import G2p
from .en_normalization import normalize
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag

# Reuse symbols_v2 from Japanese module for now (it already includes ARPAbet)
from ..Japanese.SymbolsV2 import symbols_v2, symbol_to_id_v2


_word_tokenize = TweetTokenizer().tokenize


# Align punctuation replacement with GPT-SoVITS english.py
_REP_MAP = {
    "[;:：，；]": ",",
    "[\"’]": "'",
    "。": ".",
    "！": "!",
    "？": "?",
}

_PUNCTS = ",[.?!…]"


def _replace_consecutive_punctuation(text: str) -> str:
    punct = ",[.?!…]"
    pattern = f"([{punct}])([{punct}])+"
    return re.sub(pattern, r"\\1", text)


def text_normalize(text: str) -> str:
    pattern = re.compile("|".join(re.escape(k) for k in _REP_MAP.keys()))
    text = pattern.sub(lambda x: _REP_MAP[x.group()], text)
    text = _replace_consecutive_punctuation(text)
    return text


class _EN_G2P:
    def __init__(self):
        self._g2p = G2p()

    def __call__(self, text: str) -> List[str]:
        words = _word_tokenize(text)
        tokens = pos_tag(words)
        prons: List[str] = []
        for word, _ in tokens:
            phonemes = self._g2p(word)
            prons.extend(phonemes)
            prons.append(" ")
        return prons[:-1]


_g2p = _EN_G2P()


def english_to_phones(text: str) -> List[int]:
    text = text_normalize(text)
    text = normalize(text)
    phone_list = _g2p(text)
    # Filter unknowns and non-symbols. Map to IDs via symbols_v2.
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]
    phones = [ph for ph in phones if ph in symbols_v2]
    # Stability: if too short, prepend comma
    if len(phones) < 4:
        phones = [","] + phones
    return [symbol_to_id_v2[ph] for ph in phones]


