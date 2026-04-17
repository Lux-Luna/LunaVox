"""Unit tests for :mod:`lunavox.core.text.streaming`.

Migrated and extended from the former ``test_serve_sentence_buffer``
suite. The buffer is now package-neutral (lives under
:mod:`lunavox.core.text`) so it doesn't need the ``[serve]`` extra.
"""

from __future__ import annotations

from lunavox.core.text import StreamingSentenceBuffer, split_into_sentences

# ---------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------


def test_split_western_punctuation():
    text = "Hello world. How are you? I am fine!"
    out = split_into_sentences(text)
    assert len(out) == 3
    assert "Hello world." in out[0]
    assert "How are you?" in out[1]
    assert "I am fine!" in out[2]


def test_split_cjk_punctuation():
    text = "你好。今天天气怎么样？我很好！"
    out = split_into_sentences(text)
    assert len(out) == 3
    assert "你好" in out[0]
    assert "今天天气怎么样" in out[1]
    assert "我很好" in out[2]


def test_split_mixed_scripts():
    """Mixed CJK + ASCII text shares the same algorithm: each tier's
    own terminator fires independently, and we recover all three
    sentences without per-language branches."""
    text = "Hello world. 你好世界。How are you?"
    out = split_into_sentences(text)
    assert len(out) == 3


def test_split_empty_input_returns_empty_list():
    assert split_into_sentences("") == []


# ---------------------------------------------------------------------
# StreamingSentenceBuffer
# ---------------------------------------------------------------------


def test_buffer_emits_only_complete_sentences():
    buf = StreamingSentenceBuffer()
    buf.feed("Hello ")
    assert buf.drain() == []
    buf.feed("world. How are ")
    sentences = buf.drain()
    assert len(sentences) == 1
    assert "Hello world" in sentences[0]
    assert buf.drain() == []


def test_buffer_handles_multiple_sentences_in_one_feed():
    buf = StreamingSentenceBuffer()
    buf.feed("First sentence. Second sentence! Third one? ")
    assert len(buf.drain()) == 3


def test_buffer_flush_returns_partial_tail():
    buf = StreamingSentenceBuffer()
    buf.feed("Complete one. Partial two")
    assert len(buf.drain()) == 1
    leftover = buf.flush()
    assert len(leftover) == 1
    assert "Partial two" in leftover[0]


def test_buffer_flush_when_empty_returns_nothing():
    buf = StreamingSentenceBuffer()
    assert buf.flush() == []
    buf.feed("done. ")
    buf.drain()
    assert buf.flush() == []


def test_short_fragment_is_held_back():
    """Abbreviation-style 'Mr.' fragments shouldn't emit as standalone
    sentences — the 4-char minimum keeps them queued."""
    buf = StreamingSentenceBuffer()
    buf.feed("Mr. ")
    assert buf.drain() == []


def test_buffer_handles_cjk_streaming():
    """CJK punctuation is self-terminating — no trailing space needed."""
    buf = StreamingSentenceBuffer()
    buf.feed("你好世界。")
    sentences = buf.drain()
    assert len(sentences) == 1
    assert "你好世界" in sentences[0]


def test_buffer_handles_japanese_streaming():
    buf = StreamingSentenceBuffer()
    buf.feed("これは最初の文章です。")
    sentences = buf.drain()
    assert len(sentences) == 1


def test_buffer_token_by_token_feed():
    """Simulate an LLM streaming one token at a time."""
    buf = StreamingSentenceBuffer()
    emitted: list[str] = []
    for token in ["Hello", " world", ".", " Next", " one", "!", " "]:
        buf.feed(token)
        emitted.extend(buf.drain())
    # After the final space, both sentences should be out.
    emitted.extend(buf.flush())
    joined = " ".join(emitted)
    assert "Hello world" in joined
    assert "Next one" in joined
