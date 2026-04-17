"""Unit tests for :class:`lunavox.core.text.splitter.TextSplitter`.

Covers each stage of the cascade (strong → weak → soft → hard), each
of the 10 languages the LunaVox model supports, and the short-
segment merge pass.
"""

from __future__ import annotations

import pytest

from lunavox.core.text.splitter import TextSplitter

# ---------------------------------------------------------------------
# fast path: short inputs bypass the splitter
# ---------------------------------------------------------------------


def test_short_input_returns_single_segment():
    s = TextSplitter(max_chars=100)
    assert s.split("Hello world.") == ["Hello world."]


def test_empty_input_returns_empty_list():
    s = TextSplitter()
    assert s.split("") == []
    assert s.split("   \n\t ") == []


def test_constructor_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        TextSplitter(max_chars=0)
    with pytest.raises(ValueError):
        TextSplitter(max_chars=10, min_chars=0)
    with pytest.raises(ValueError):
        TextSplitter(max_chars=10, min_chars=20)


# ---------------------------------------------------------------------
# strong terminator pass (stage 1)
# ---------------------------------------------------------------------


def test_strong_split_on_english():
    # Well over max_chars to force real splitting.
    text = "First sentence here. Second sentence is also here! And a third one? Finally the fourth."
    s = TextSplitter(max_chars=40, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 3
    joined = " ".join(segments)
    for word in ("First", "Second", "third", "fourth"):
        assert word in joined


def test_strong_split_on_chinese_fullwidth():
    text = "第一句很长需要被切分。第二句也是一个完整的句子！第三句是一个疑问句？第四句是个终结句。"
    s = TextSplitter(max_chars=16, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 3
    for marker in ("第一句", "第二句", "第三句", "第四句"):
        assert any(marker in seg for seg in segments)


def test_strong_split_on_japanese():
    text = "これは最初の文章です。これは二つ目の文章ですよ！最後の質問はありますか？"
    s = TextSplitter(max_chars=20, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2
    assert any("最初" in seg for seg in segments)
    assert any("最後" in seg for seg in segments)


def test_strong_split_on_russian_cyrillic():
    text = (
        "Это первое предложение на русском. "
        "Это второе предложение, тоже на русском! "
        "А это третий вопрос?"
    )
    s = TextSplitter(max_chars=40, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2
    joined = " ".join(segments)
    assert "первое" in joined and "третий" in joined


def test_strong_split_on_korean_with_ascii_terminators():
    text = "안녕하세요. 오늘 날씨가 어때요? 좋은 하루 보내세요!"
    s = TextSplitter(max_chars=18, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2


def test_strong_split_preserves_all_content():
    """Rejoining the segments (with spaces where they were) must
    recover every word of the original — no silent drops."""
    text = "Sentence one. Sentence two. Sentence three."
    s = TextSplitter(max_chars=20, min_chars=4)
    segments = s.split(text)
    joined = " ".join(segments).replace("  ", " ")
    for word in ("one", "two", "three"):
        assert word in joined


# ---------------------------------------------------------------------
# weak terminator fallback (stage 2)
# ---------------------------------------------------------------------


def test_weak_split_kicks_in_when_strong_chunks_still_too_long():
    # One giant "strong" segment (no ./!/?) with commas inside.
    text = "alpha, beta, gamma, delta, epsilon, zeta, eta, theta"
    s = TextSplitter(max_chars=16, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2
    for tag in ("alpha", "beta", "zeta", "theta"):
        assert any(tag in seg for seg in segments)


def test_weak_split_chinese_commas():
    text = "第一部分内容非常非常长，第二部分内容也是如此长，第三部分也很长，第四部分到此结束"
    s = TextSplitter(max_chars=20, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2


# ---------------------------------------------------------------------
# soft break fallback (stage 3)
# ---------------------------------------------------------------------


def test_soft_break_splits_long_unpunctuated_western():
    text = "word " * 80  # 400 chars, no punctuation at all
    s = TextSplitter(max_chars=50, min_chars=10)
    segments = s.split(text.strip())
    assert len(segments) >= 2
    assert all(len(seg) <= 50 for seg in segments)


# ---------------------------------------------------------------------
# hard cut fallback (stage 4)
# ---------------------------------------------------------------------


def test_hard_cut_on_unpunctuated_cjk_run():
    # Pure CJK without any punctuation nor whitespace — only hard cut can split this.
    text = "中" * 200
    s = TextSplitter(max_chars=50, min_chars=4)
    segments = s.split(text)
    assert len(segments) == 4
    assert all(len(seg) == 50 for seg in segments)
    assert "".join(segments) == text


def test_hard_cut_on_urlish_run():
    text = "https://example.com/" + "a" * 200
    s = TextSplitter(max_chars=40, min_chars=10)
    segments = s.split(text)
    assert all(len(seg) <= 40 for seg in segments)
    # All content preserved (join reconstruction, stripped of any trailing
    # whitespace that the cascade may have trimmed).
    assert "".join(segments) == text


# ---------------------------------------------------------------------
# max_chars invariant
# ---------------------------------------------------------------------


def test_no_segment_exceeds_max_chars_regardless_of_input():
    inputs = [
        "Normal English sentences. With punctuation. And more.",
        "中文的段落。" * 50,
        "a" * 500,
        "word " * 100,
        "日本語の長い文章。" * 30,
        "alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, " * 5,
    ]
    s = TextSplitter(max_chars=60, min_chars=4)
    for text in inputs:
        segments = s.split(text)
        assert all(len(seg) <= s.max_chars for seg in segments), (
            f"max_chars violated on input: {text[:40]!r}"
        )


# ---------------------------------------------------------------------
# short-segment merge
# ---------------------------------------------------------------------


def test_short_fragments_get_merged_when_safe():
    # Three tiny sentences whose combined length fits in max_chars.
    text = "Hi. Yo. Hey."
    s = TextSplitter(max_chars=40, min_chars=8)
    segments = s.split(text)
    # The result should be one merged segment, not three fragments.
    assert len(segments) == 1
    assert "Hi" in segments[0] and "Yo" in segments[0] and "Hey" in segments[0]


def test_short_fragments_not_merged_when_it_would_overflow():
    text = "Hi. " + "x" * 35 + ". " + "y" * 5 + "."
    s = TextSplitter(max_chars=40, min_chars=10)
    segments = s.split(text)
    assert all(len(seg) <= 40 for seg in segments)
    # Specifically: "Hi." shouldn't get glued onto the long segment.
    assert any(seg.strip().startswith("Hi") for seg in segments)


# ---------------------------------------------------------------------
# european languages (coverage smoke)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Das ist der erste Satz. Hier kommt der zweite! Und eine Frage?",
        "C'est la première phrase. Voici la deuxième! Une question?",
        "Questa è la prima frase. Ecco la seconda! Una domanda?",
        "Esta es la primera oración. Aquí está la segunda! Una pregunta?",
        "Esta é a primeira frase. Aqui está a segunda! Uma pergunta?",
    ],
)
def test_european_languages_split_cleanly(text):
    """DE / FR / IT / ES / PT should all split on standard Western
    punctuation — we share the same STRONG_TERMINATORS_TRAIL table."""
    s = TextSplitter(max_chars=32, min_chars=4)
    segments = s.split(text)
    assert len(segments) >= 2
