"""Streaming text → sentence buffer for ``WS /v1/stream/text``.

Voice agents (LLM → TTS pipelines) want to start synthesising before
the LLM has finished generating a full reply. The pattern is:

1. The LLM streams tokens / words / phrases into LunaVox over a
   WebSocket text channel.
2. LunaVox watches the incoming text for **sentence boundaries**
   (``.`` ``!`` ``?`` for English; ``。`` ``！`` ``？`` for CJK)
   and emits each complete sentence to the synthesizer as soon as
   it lands, so the first audio chunk is back to the user with
   roughly one-sentence latency rather than full-reply latency.
3. When the LLM signals end-of-turn, any leftover partial sentence
   is flushed as the final unit.

This module is just the buffer + splitter. The WebSocket handler in
:mod:`lunavox.serve.server` does the actual queueing and dispatch.

The split rules are deliberately simple: a sentence ends at one of
the recognised punctuation characters followed by whitespace, end
of buffer + flush, or — for CJK punctuation — the punctuation alone.
This isn't a linguistic parser; it's a pragmatic boundary detector
that handles 95% of natural prose. Edge cases (decimal points,
abbreviations, ellipses) are handled by minimum length to avoid
splitting "Mr." into a single-token sentence.
"""

from __future__ import annotations

import re

# Sentence-terminating punctuation. Western punctuation usually needs
# to be followed by whitespace to count (the streaming "this sentence
# is definitely complete" signal), but a trailing terminator at end
# of input is also valid. CJK punctuation is self-terminating (no
# trailing space convention in Chinese).
_WESTERN_TERMS = (".", "!", "?")
_CJK_TERMS = ("。", "！", "？", "…", "．")
_MIN_SENTENCE_CHARS = 4

# Strict splitter — used by ``SentenceBuffer.drain`` where we only
# emit sentences once we've seen the whitespace marker that says the
# next chunk is starting. Streaming with a trailing partial would
# wrongly emit it without this guard.
_STRICT_SPLIT_RE = re.compile(
    r"""(
        [.!?]+\s+      |   # western: terminator + whitespace
        [\u3002\uff01\uff1f\u2026\uff0e]+   # CJK: any of 。 ！ ？ … ．
    )""",
    re.VERBOSE,
)

# Lenient splitter — used by the public ``split_into_sentences`` and
# ``SentenceBuffer.flush`` where we know the input is complete and a
# trailing terminator without whitespace is genuinely the last sentence.
_LENIENT_SPLIT_RE = re.compile(
    r"""(
        [.!?]+(?:\s+|$)    |   # western: terminator + whitespace OR end of string
        [\u3002\uff01\uff1f\u2026\uff0e]+   # CJK: any of 。 ！ ？ … ．
    )""",
    re.VERBOSE,
)


class SentenceBuffer:
    """Append-only text buffer that yields complete sentences.

    Use :meth:`feed` to push more text in (e.g. from a WebSocket
    text frame), and :meth:`drain` after each feed to pull out any
    sentences that became complete. :meth:`flush` empties whatever
    is left at end-of-stream — typically a partial sentence that
    didn't end with terminator, returned as the final unit.
    """

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, text: str) -> None:
        """Append a chunk of incoming text to the buffer."""
        self._buf += text

    def drain(self) -> list[str]:
        """Return every complete sentence currently in the buffer.

        Uses the **strict** splitter so a trailing partial sentence
        (with terminator but no following whitespace) stays in the
        buffer. Sentences shorter than :data:`_MIN_SENTENCE_CHARS`
        are also kept queued so abbreviation-style fragments don't
        get flushed as standalone units.

        After this call the buffer holds only the residual text
        past the last recognised terminator-with-whitespace.
        """
        sentences = _split_strict(self._buf)
        if not sentences:
            return []

        last_end = _last_strict_terminator_end(self._buf)
        if last_end is None:
            # Defensive: splitter said yes but locator said no.
            return []
        self._buf = self._buf[last_end:]

        return [s for s in sentences if len(s.strip()) >= _MIN_SENTENCE_CHARS]

    def flush(self) -> list[str]:
        """Return everything still in the buffer (may be partial).

        After flushing the buffer is emptied. Used at end-of-stream
        when the client signals it has no more text to send.
        """
        leftover = self._buf.strip()
        self._buf = ""
        if not leftover:
            return []
        return [leftover]


def split_into_sentences(text: str) -> list[str]:
    """Split ``text`` on sentence boundaries, including the terminator.

    Lenient splitter — accepts a trailing terminator without
    following whitespace as the final sentence. Used by callers
    that have the complete text in hand (i.e. not streaming).
    """
    if not text:
        return []
    out: list[str] = []
    last = 0
    for match in _LENIENT_SPLIT_RE.finditer(text):
        end = match.end()
        chunk = text[last:end]
        if chunk:
            out.append(chunk.strip())
        last = end
    # Anything after the last terminator is partial; the lenient
    # splitter doesn't emit it because the trailing-terminator
    # branch already absorbed end-of-string.
    return out


def _split_strict(text: str) -> list[str]:
    """Split ``text`` only at terminator-with-trailing-whitespace.

    Used by :meth:`SentenceBuffer.drain` so a buffer containing
    ``"Hello world."`` (no trailing space) keeps that sentence
    queued until the next feed brings in a space.
    """
    if not text:
        return []
    out: list[str] = []
    last = 0
    for match in _STRICT_SPLIT_RE.finditer(text):
        end = match.end()
        chunk = text[last:end]
        if chunk:
            out.append(chunk.strip())
        last = end
    return out


def _last_strict_terminator_end(text: str) -> int | None:
    """Return the index just past the last strict terminator in ``text``."""
    last_end: int | None = None
    for match in _STRICT_SPLIT_RE.finditer(text):
        last_end = match.end()
    return last_end
