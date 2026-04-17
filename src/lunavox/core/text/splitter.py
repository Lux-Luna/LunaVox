"""Batch text splitter — "give me a long string, get a list of segments".

Used by :class:`~lunavox.core.synth.pipeline.SynthesisPipeline` to
pre-chunk long inputs before sending each segment to the engine.
Distinct from :class:`~lunavox.core.text.streaming.StreamingSentenceBuffer`
(which handles live token feeds): this one receives the whole text
at once and must decide boundaries up-front.

The algorithm is a **single cascading pass** that never branches on
language. Every decision consults the tables in
:mod:`~lunavox.core.text.punctuation`, so adding a writing system is
a data-only change.

Algorithm:

1. Emit candidate splits at every strong terminator (sentence end).
2. For any segment still longer than ``max_chars``, re-split it at
   weak terminators (clause end).
3. For any segment *still* too long, re-split at whitespace.
4. For any segment *still* too long (long unpunctuated CJK or URL-
   like strings), hard-cut at exactly ``max_chars``.

After step 1, short adjacent segments are greedily merged back
together so no output is shorter than ``min_chars`` unless it's the
last remainder — this avoids flooding the engine with 3-character
fragments for very spiky punctuation density.
"""

from __future__ import annotations

import re
from typing import Optional

from .punctuation import (
    SOFT_BREAK,
    STRONG_TERMINATORS_SELF,
    STRONG_TERMINATORS_TRAIL,
    WEAK_TERMINATORS_SELF,
    WEAK_TERMINATORS_TRAIL,
)


def _build_boundary_re(self_terms: str, trail_terms: str) -> re.Pattern[str]:
    """Build a regex that matches one-or-more terminators plus any
    trailing whitespace, for the given tier.

    Self-terminating chars match on their own. Trail-terminating chars
    only match when followed by whitespace or end-of-input.
    """
    self_cls = re.escape(self_terms)
    trail_cls = re.escape(trail_terms)
    # A boundary is:
    #   - one-or-more self-terminators (greedy, to eat "。。。"), then optional space, OR
    #   - one-or-more trail-terminators followed by whitespace or end-of-string
    return re.compile(
        rf"(?:[{self_cls}]+\s*)|(?:[{trail_cls}]+(?=\s|$))[{trail_cls}\s]*",
    )


_STRONG_BOUNDARY_RE: re.Pattern[str] = _build_boundary_re(
    STRONG_TERMINATORS_SELF, STRONG_TERMINATORS_TRAIL
)
_WEAK_BOUNDARY_RE: re.Pattern[str] = _build_boundary_re(
    WEAK_TERMINATORS_SELF, WEAK_TERMINATORS_TRAIL
)
_SOFT_BREAK_RE: re.Pattern[str] = re.compile(f"[{re.escape(SOFT_BREAK)}]+")


def _split_by_regex(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split ``text`` at every match of ``pattern``, keeping the
    terminator glued onto the preceding segment.

    Unlike ``re.split`` with a capture group we keep whitespace that
    belonged to the separator so rejoining the list reproduces the
    original text exactly — useful for round-trip tests.
    """
    if not text:
        return []
    out: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        end = match.end()
        piece = text[last:end]
        if piece:
            out.append(piece)
        last = end
    tail = text[last:]
    if tail:
        out.append(tail)
    return out


class TextSplitter:
    """Split long text into synth-ready segments.

    Parameters:

    * ``max_chars`` — hard upper bound on any emitted segment. Default
      240 is roughly 10–20 seconds of audio depending on language
      density; tuned to stay well under typical ``max_audio_tokens``
      limits while keeping streaming latency per segment low.
    * ``min_chars`` — floor on segment length when merging short
      pieces. Avoids flooding the engine with standalone fragments
      like "Yes." or "了。" whose synthesis overhead dominates.
    * ``prefer_strong`` — kept for future use; today the cascade
      already prefers strong → weak → soft, so this toggle is
      primarily documentation.
    * ``language_hint`` — ignored by the default implementation (the
      tables cover every supported language). Reserved so future
      optional plugins (e.g. ``pysbd``) can specialise per-language
      without changing the call site.
    """

    def __init__(
        self,
        *,
        max_chars: int = 240,
        min_chars: int = 12,
        prefer_strong: bool = True,
        language_hint: Optional[str] = None,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if min_chars < 1 or min_chars > max_chars:
            raise ValueError(
                f"min_chars must be in [1, max_chars={max_chars}], got {min_chars}"
            )
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.prefer_strong = prefer_strong
        self.language_hint = language_hint

    # --- public API ---------------------------------------------------

    def split(self, text: str) -> list[str]:
        """Chunk ``text`` into segments honoring the max/min bounds.

        Always returns at least one segment (the whole input
        stripped) unless the input is empty or all whitespace, in
        which case an empty list is returned.
        """
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= self.max_chars:
            # Fast path: already short enough, no splitting needed.
            return [stripped]

        # Pass 1: strong terminators.
        segments = _split_by_regex(text, _STRONG_BOUNDARY_RE)
        # Pass 2: weak terminators on any still-too-long segment.
        segments = self._refine(segments, _WEAK_BOUNDARY_RE)
        # Pass 3: whitespace on any still-too-long segment.
        segments = self._refine(segments, _SOFT_BREAK_RE)
        # Pass 4: hard cut on anything that still exceeds max_chars
        #         (e.g. URL-like runs, long CJK with no punctuation).
        segments = self._hard_cut(segments)
        # Merge short neighbors so no segment is shorter than min_chars
        # (last segment is exempt — it's always a valid remainder).
        segments = self._merge_short(segments)
        # Trim whitespace — callers want clean synth input; the raw
        # terminator+space gluing was only needed to maintain sound
        # rejoin semantics during the cascade.
        return [s.strip() for s in segments if s.strip()]

    # --- internals ----------------------------------------------------

    def _refine(
        self, segments: list[str], pattern: re.Pattern[str]
    ) -> list[str]:
        """Re-split any segment longer than ``max_chars`` using ``pattern``."""
        out: list[str] = []
        for seg in segments:
            if len(seg) <= self.max_chars:
                out.append(seg)
                continue
            parts = _split_by_regex(seg, pattern)
            if len(parts) <= 1:
                # Pattern didn't match inside — leave untouched for the
                # next pass.
                out.append(seg)
            else:
                out.extend(parts)
        return out

    def _hard_cut(self, segments: list[str]) -> list[str]:
        """Slice any still-too-long segment at exactly ``max_chars``."""
        out: list[str] = []
        for seg in segments:
            if len(seg) <= self.max_chars:
                out.append(seg)
                continue
            start = 0
            while start < len(seg):
                out.append(seg[start : start + self.max_chars])
                start += self.max_chars
        return out

    def _merge_short(self, segments: list[str]) -> list[str]:
        """Greedy-merge short neighbors until each hits ``min_chars``.

        Merging only happens when the combined length still fits in
        ``max_chars``. If merging would overflow, the short segment is
        emitted as-is — better short than over-limit.
        """
        if not segments:
            return segments
        out: list[str] = []
        buf = ""
        for seg in segments:
            if not buf:
                buf = seg
                continue
            if len(buf.strip()) < self.min_chars and len(buf) + len(seg) <= self.max_chars:
                buf += seg
            else:
                out.append(buf)
                buf = seg
        if buf:
            out.append(buf)
        return out
