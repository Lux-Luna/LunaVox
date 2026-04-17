"""Multi-language text utilities used by the synthesis pipeline.

Two collaborating types live here:

* :class:`TextSplitter` — chunks a long string at sentence boundaries,
  cascading through punctuation strength tiers down to a hard cut so
  no segment exceeds ``max_chars``. Used by the synth pipeline to
  pre-split long inputs before each segment goes to the engine.
* :class:`StreamingSentenceBuffer` — append-only buffer for live
  token streams (LLM → TTS), emitting complete sentences as soon as
  a terminator arrives. Used by ``WS /v1/stream/text``.

Both share the punctuation tables in :mod:`punctuation`, so adding a
language is a one-file change to the data — no per-language code
branches anywhere in the pipeline.
"""

from __future__ import annotations

from .splitter import TextSplitter
from .streaming import StreamingSentenceBuffer, split_into_sentences

__all__ = [
    "StreamingSentenceBuffer",
    "TextSplitter",
    "split_into_sentences",
]
