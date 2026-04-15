"""Unit tests for :mod:`lunavox.serve.sentence_buffer`.

Pure Python — no FastAPI / pydantic / engine dependency. Skip
pattern matches the rest of the serve test suite for consistency.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pydantic",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)


def test_split_western_punctuation():
    from lunavox.serve.sentence_buffer import split_into_sentences

    text = "Hello world. How are you? I am fine!"
    out = split_into_sentences(text)
    assert len(out) == 3
    assert "Hello world." in out[0]
    assert "How are you?" in out[1]
    assert "I am fine!" in out[2]


def test_split_cjk_punctuation():
    from lunavox.serve.sentence_buffer import split_into_sentences

    text = "你好。今天天气怎么样？我很好！"
    out = split_into_sentences(text)
    assert len(out) == 3
    assert "你好" in out[0]
    assert "今天天气怎么样" in out[1]
    assert "我很好" in out[2]


def test_buffer_emits_only_complete_sentences():
    from lunavox.serve.sentence_buffer import SentenceBuffer

    buf = SentenceBuffer()
    buf.feed("Hello ")
    assert buf.drain() == []  # no terminator yet
    buf.feed("world. How are ")
    sentences = buf.drain()
    assert len(sentences) == 1
    assert "Hello world" in sentences[0]
    # The unfinished "How are" stays in the buffer
    assert buf.drain() == []


def test_buffer_handles_multiple_sentences_in_one_feed():
    from lunavox.serve.sentence_buffer import SentenceBuffer

    buf = SentenceBuffer()
    buf.feed("First sentence. Second sentence! Third one? ")
    sentences = buf.drain()
    assert len(sentences) == 3


def test_buffer_flush_returns_partial_tail():
    from lunavox.serve.sentence_buffer import SentenceBuffer

    buf = SentenceBuffer()
    buf.feed("Complete one. Partial two")
    sentences = buf.drain()
    assert len(sentences) == 1
    leftover = buf.flush()
    assert len(leftover) == 1
    assert "Partial two" in leftover[0]


def test_buffer_flush_when_empty_returns_nothing():
    from lunavox.serve.sentence_buffer import SentenceBuffer

    buf = SentenceBuffer()
    assert buf.flush() == []
    buf.feed("done. ")
    buf.drain()
    assert buf.flush() == []  # already drained


def test_short_fragment_is_held_back():
    """Abbreviation-style 'Mr.' fragments shouldn't be emitted as
    standalone sentences — the minimum-length filter keeps them
    queued until enough text accumulates."""
    from lunavox.serve.sentence_buffer import SentenceBuffer

    buf = SentenceBuffer()
    buf.feed("Mr. ")
    sentences = buf.drain()
    # Either empty (preferred — fragment held) or empty after filter.
    assert sentences == []
