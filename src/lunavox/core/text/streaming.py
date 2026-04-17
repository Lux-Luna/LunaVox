"""Streaming sentence buffer for live LLM → TTS token pipelines.

Paired with :class:`~lunavox.core.text.splitter.TextSplitter` but
solves a different problem: the splitter receives the whole text
up-front and must choose boundaries; this buffer receives text
incrementally and must emit a complete sentence as soon as one
lands so the synthesizer can start generating audio with a one-
sentence latency instead of a full-reply latency.

Used by ``WS /v1/stream/text`` to watch LLM output for sentence
boundaries. The split rules are deliberately pragmatic:

* A sentence ends at any **self-terminating** strong punctuation
  (``。 ！ ？ …``) or at a **trail-terminating** ASCII terminator
  (``. ! ?``) followed by whitespace. The trailing whitespace
  requirement avoids slicing mid-abbreviation ("Mr.") or decimal
  ("3.14") in streaming mode — the next feed will bring the space
  that confirms the sentence really ended.
* :meth:`flush` relaxes that rule — at end-of-stream we know no
  more text is coming, so a trailing terminator without whitespace
  is genuinely the final sentence.
* Sentences shorter than :data:`_MIN_SENTENCE_CHARS` are held back
  and merged with the next one, so abbreviation fragments don't
  escape as standalone units.

The punctuation tables come from :mod:`~lunavox.core.text.punctuation`,
so a new language is a one-file data change — no code here references
specific characters.
"""

from __future__ import annotations

import re

from .punctuation import (
    STRONG_TERMINATORS_SELF,
    STRONG_TERMINATORS_TRAIL,
)

_MIN_SENTENCE_CHARS = 4


def _build_split_re(*, strict: bool) -> re.Pattern[str]:
    """Build the sentence splitter for one of the two semantics.

    * ``strict=True`` — trail-terminators require following whitespace.
      Used by :meth:`SentenceBuffer.drain` during streaming: a
      sentence with ``.`` but no trailing space stays buffered in
      case the next feed brings more text.
    * ``strict=False`` — trail-terminators also match at end-of-string.
      Used by :meth:`SentenceBuffer.flush` and :func:`split_into_sentences`,
      where we know the caller has no more text.
    """
    self_cls = re.escape(STRONG_TERMINATORS_SELF)
    trail_cls = re.escape(STRONG_TERMINATORS_TRAIL)
    trail_context = r"\s+" if strict else r"(?:\s+|$)"
    return re.compile(
        rf"""(
            [{trail_cls}]+{trail_context}    |
            [{self_cls}]+
        )""",
        re.VERBOSE,
    )


_STRICT_SPLIT_RE: re.Pattern[str] = _build_split_re(strict=True)
_LENIENT_SPLIT_RE: re.Pattern[str] = _build_split_re(strict=False)


class StreamingSentenceBuffer:
    """Append-only buffer that yields complete sentences as they arrive.

    Use :meth:`feed` to push incoming text, then :meth:`drain` after
    each feed to pull out any sentences that became complete.
    :meth:`flush` empties whatever is left at end-of-stream — typically
    a partial sentence that didn't end with a terminator.

    This class was formerly ``lunavox.serve.sentence_buffer.SentenceBuffer``;
    it has been lifted into :mod:`lunavox.core.text` so both the
    streaming-input WebSocket and any future consumer (CLI REPL, GUI
    live preview) share one implementation.
    """

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, text: str) -> None:
        """Append an incoming text fragment to the buffer."""
        self._buf += text

    def drain(self) -> list[str]:
        """Return every complete sentence currently in the buffer.

        Uses the **strict** splitter, so a trailing partial sentence
        (terminator but no following whitespace) stays queued for
        the next feed. Short fragments below :data:`_MIN_SENTENCE_CHARS`
        are filtered out so "Mr." doesn't escape as its own sentence.
        """
        sentences = _split_strict(self._buf)
        if not sentences:
            return []

        last_end = _last_strict_terminator_end(self._buf)
        if last_end is None:
            return []
        self._buf = self._buf[last_end:]

        return [s for s in sentences if len(s.strip()) >= _MIN_SENTENCE_CHARS]

    def flush(self) -> list[str]:
        """Return everything still buffered (may be partial) and clear it.

        Used at end-of-stream when the client signals no more input.
        """
        leftover = self._buf.strip()
        self._buf = ""
        if not leftover:
            return []
        return [leftover]


def split_into_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping the terminator glued on.

    Lenient splitter — treats a trailing terminator without following
    whitespace as the final sentence. Used by callers that have the
    entire text on hand (non-streaming).
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
    return out


def _split_strict(text: str) -> list[str]:
    """Split ``text`` only at terminator-with-trailing-whitespace boundaries."""
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
