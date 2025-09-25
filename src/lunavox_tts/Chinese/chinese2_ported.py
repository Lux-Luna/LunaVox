import os
import re
import logging
import cn2an
from typing import Tuple

from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

import jieba_fast
jieba_fast.setLogLevel(logging.CRITICAL)
import jieba_fast.posseg as psg

from .tone_sandhi import ToneSandhi


current_file_path = os.path.dirname(__file__)
_OPENCPOP_PATH = os.path.join(current_file_path, "opencpop_strict.txt")
_mapping_lines = []
with open(_OPENCPOP_PATH, "r", encoding="utf-8") as f:
    for raw in f.read().splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        left, *rest = line.split("\t")
        if not rest:
            continue
        right = rest[0].strip()
        if not left or not right:
            continue
        _mapping_lines.append((left, right))

pinyin_to_symbol_map = {k: v for k, v in _mapping_lines}

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

punctuation = ["!", "?", "…", ",", ".", "-"]

tone_modifier = ToneSandhi()


def replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)
    return replaced_text


def replace_consecutive_punctuation(text: str) -> str:
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text: str) -> str:
    # Minimal numeric normalization to match cn2an usage
    try:
        text = cn2an.transform(text, "an2cn")
    except Exception:
        pass
    sentences = [text]
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


def g2p(text: str) -> Tuple[list, list]:
    pattern = r"(?<=%s)\s*" % ("".join(punctuation),)
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    return _g2p(sentences)


def _get_initials_finals(word: str):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
not_erhua = {"虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿", "狗儿", "少儿"}


def _merge_erhua(initials: list, finals: list, word: str, pos: str):
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals
    if len(finals) != len(word):
        return initials, finals
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn in {"er2", "er5"} and word[-2:] not in not_erhua and new_finals:
            phn = "er" + new_finals[-1][-1]
        new_initials.append(initials[i])
        new_finals.append(phn)
    return new_initials, new_finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    for seg in segments:
        pinyins = []
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        initials = []
        finals = []

        # Use pypinyin fallback + tone sandhi with erhua; users get good baseline without ONNX builder
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
            initials.append(sub_initials)
            finals.append(sub_finals)
        initials = sum(initials, [])
        finals = sum(finals, [])

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            if c == v:
                # punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]
                pinyin = c + v_without_tone
                if c:
                    v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
                    if v_without_tone in v_rep_map:
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
                    if pinyin in pinyin_rep_map:
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                        if pinyin and pinyin[0] in single_rep_map:
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
                if pinyin not in pinyin_to_symbol_map:
                    phone = ["UNK"]
                    word2ph.append(1)
                else:
                    new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                    new_v = new_v + tone
                    phone = [new_c, new_v]
                    word2ph.append(len(phone))
            phones_list += phone
    return phones_list, word2ph


