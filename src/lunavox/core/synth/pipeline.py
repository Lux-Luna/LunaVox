"""Synthesis pipelines — the unified entry every frontend calls into.

Two pipelines live here, parameterised by the engine they wrap:

* :class:`SynthesisPipeline` — sync, takes one :class:`Engine`.
  Used by CLI (one-shot) and GUI (background thread).
* :class:`AsyncSynthesisPipeline` — async, takes one
  :class:`BatchEngine`. Used by the FastAPI handlers so the event
  loop stays responsive while synthesis runs under back-pressure
  from the engine pool.

Both add exactly three things on top of the raw engine:

1. **Long-text auto-splitting** — inputs over ``auto_split_threshold``
   chars are chopped by :class:`TextSplitter` and synthesized segment
   by segment. Output is concatenated (non-streaming) or flattened
   into one stream (streaming).
2. **Terminal-chunk discipline** — when joining N stream segments,
   only the *final* segment's last chunk carries ``is_last=True``.
   Intermediate ``is_last`` markers are rewritten to ``False`` so
   downstream consumers see one clean end-of-stream signal.
3. **Consistent default surface** — ``voice`` / ``params`` default
   to ``Voice.base()`` and ``SynthesisParams()`` so callers can
   always pass ``None`` through without guard clauses.

No entry should ever talk to ``Engine.synthesize(_stream)`` directly
once the migration lands — the pipeline is the contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

from lunavox.runtime import (
    BatchEngine,
    Engine,
    MemStats,
    SynthesisChunk,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
    Voice,
)

from ..text import TextSplitter


def _merge_audio(results: list[SynthesisResult]) -> Any:
    """Concatenate per-segment float32 audio arrays into one."""
    import numpy as np

    if not results:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([r.audio for r in results]).astype(np.float32, copy=False)


def _aggregate_stats(stats_list: list[SynthesisStats]) -> SynthesisStats:
    """Sum / max stats across segments so the caller sees one total.

    Timings (``t_*_ms``, ``audio_duration_ms``) are summed; memory peaks
    take the max across segments (a peak is a supremum, not a total);
    the ``start`` markers take the *min* (the earliest anchor across
    segments) so ``peak - start`` stays a meaningful high-water delta
    for the whole session. ``rtf`` is recomputed from cumulative
    compute / audio; end-of-run byte counts take the final segment.
    """
    if not stats_list:
        return SynthesisStats()
    total_compute = sum(s.t_total_ms for s in stats_list)
    total_audio = sum(s.audio_duration_ms for s in stats_list)
    rtf = (total_compute / total_audio) if total_audio > 0 else 0.0
    last_mem = stats_list[-1].mem
    # `vram_measured` aggregates conservatively: true only if every
    # segment actually produced a real NVML reading. A single "—" segment
    # must not get silently hidden behind an averaged-looking number.
    mem = MemStats(
        rss_start_bytes=min(s.mem.rss_start_bytes for s in stats_list),
        rss_end_bytes=last_mem.rss_end_bytes,
        rss_peak_bytes=max(s.mem.rss_peak_bytes for s in stats_list),
        vram_start_bytes=min(s.mem.vram_start_bytes for s in stats_list),
        vram_end_bytes=last_mem.vram_end_bytes,
        vram_peak_bytes=max(s.mem.vram_peak_bytes for s in stats_list),
        vram_measured=all(s.mem.vram_measured for s in stats_list),
    )
    return SynthesisStats(
        t_tokenize_ms=sum(s.t_tokenize_ms for s in stats_list),
        t_encode_ms=sum(s.t_encode_ms for s in stats_list),
        t_generate_ms=sum(s.t_generate_ms for s in stats_list),
        t_decode_ms=sum(s.t_decode_ms for s in stats_list),
        t_total_ms=total_compute,
        audio_duration_ms=total_audio,
        rtf=rtf,
        mem=mem,
    )


class _PipelineBase:
    """Shared splitting policy for both sync and async pipelines.

    Kept private; callers instantiate :class:`SynthesisPipeline` or
    :class:`AsyncSynthesisPipeline`.
    """

    def __init__(
        self,
        *,
        splitter: Optional[TextSplitter] = None,
        auto_split_threshold: int = 240,
    ) -> None:
        if auto_split_threshold < 1:
            raise ValueError("auto_split_threshold must be >= 1")
        self.auto_split_threshold = auto_split_threshold
        self.splitter = (
            splitter if splitter is not None else TextSplitter(max_chars=auto_split_threshold)
        )

    def _segments(self, text: str) -> list[str]:
        """Return the segment list for ``text``.

        Short-circuits when ``len(text) <= auto_split_threshold`` so
        the fast path doesn't allocate a splitter run per call. The
        returned list is always at least length 1 unless the input
        is empty/whitespace, in which case it is length 0 and the
        caller should propagate the empty result.
        """
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= self.auto_split_threshold:
            return [stripped]
        return self.splitter.split(text)


class SynthesisPipeline(_PipelineBase):
    """Sync synthesis facade wrapping one :class:`Engine`.

    Thread-safe at the level of the wrapped engine (which is not —
    ``Engine`` serialises internally). For the serve layer, use
    :class:`AsyncSynthesisPipeline` instead.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        splitter: Optional[TextSplitter] = None,
        auto_split_threshold: int = 240,
    ) -> None:
        super().__init__(splitter=splitter, auto_split_threshold=auto_split_threshold)
        self.engine = engine

    def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        """Synthesize ``text`` in one call, auto-splitting if long.

        Short inputs hit the engine once — zero overhead. Long inputs
        are split, each segment is synthesized, and the resulting
        audio is concatenated; per-segment stats are aggregated into
        one cumulative :class:`SynthesisStats`.
        """
        segments = self._segments(text)
        if not segments:
            raise ValueError("synthesize called with empty text")
        if len(segments) == 1:
            return self.engine.synthesize(segments[0], voice=voice, params=params)

        results = [self.engine.synthesize(seg, voice=voice, params=params) for seg in segments]
        merged_audio = _merge_audio(results)
        return SynthesisResult(
            audio=merged_audio,
            sample_rate=results[0].sample_rate,
            stats=_aggregate_stats([r.stats for r in results]),
            mode=results[0].mode,
        )

    def synthesize_stream(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> Iterator[SynthesisChunk]:
        """Stream PCM chunks, auto-splitting long inputs.

        Yields chunks in order across every segment. Only the very
        last chunk of the last segment has ``is_last=True``;
        intermediate segments' terminal chunks are re-flagged as
        non-terminal so downstream consumers see exactly one end-of-
        stream signal per call. ``stats`` on the final chunk is
        :func:`_aggregate_stats` over all per-segment stats.
        """
        segments = self._segments(text)
        if not segments:
            raise ValueError("synthesize_stream called with empty text")

        collected_stats: list[SynthesisStats] = []
        for seg_idx, seg in enumerate(segments):
            is_final_segment = seg_idx == len(segments) - 1
            for chunk in self.engine.synthesize_stream(seg, voice=voice, params=params):
                if chunk.is_last:
                    if chunk.stats is not None:
                        collected_stats.append(chunk.stats)
                    if is_final_segment:
                        yield SynthesisChunk(
                            audio=chunk.audio,
                            sample_rate=chunk.sample_rate,
                            is_last=True,
                            stats=_aggregate_stats(collected_stats),
                        )
                    else:
                        # Intermediate segment's tail: demote to non-terminal
                        # so the consumer only sees one is_last=True.
                        if len(chunk.audio) > 0:
                            yield SynthesisChunk(
                                audio=chunk.audio,
                                sample_rate=chunk.sample_rate,
                                is_last=False,
                                stats=None,
                            )
                else:
                    yield chunk


class AsyncSynthesisPipeline(_PipelineBase):
    """Async synthesis facade wrapping one :class:`BatchEngine`.

    Suitable for the FastAPI layer: each call awaits on the engine
    pool's idle queue, so concurrent requests naturally back-pressure.

    For long-text inputs, each segment is submitted as an independent
    pool request — different engines may serve different segments of
    the same request, which is fine because segments are independent
    (no shared decoder state needed between them). This preserves
    pool fairness for other concurrent callers.
    """

    def __init__(
        self,
        batch: BatchEngine,
        *,
        splitter: Optional[TextSplitter] = None,
        auto_split_threshold: int = 240,
    ) -> None:
        super().__init__(splitter=splitter, auto_split_threshold=auto_split_threshold)
        self.batch = batch

    @property
    def sample_rate(self) -> int:
        return self.batch.sample_rate

    async def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        """Async one-shot synthesis with auto-split."""
        segments = self._segments(text)
        if not segments:
            raise ValueError("synthesize called with empty text")
        if len(segments) == 1:
            return await self.batch.submit(segments[0], voice=voice, params=params)

        results: list[SynthesisResult] = []
        for seg in segments:
            results.append(await self.batch.submit(seg, voice=voice, params=params))
        return SynthesisResult(
            audio=_merge_audio(results),
            sample_rate=results[0].sample_rate,
            stats=_aggregate_stats([r.stats for r in results]),
            mode=results[0].mode,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> AsyncIterator[SynthesisChunk]:
        """Async streaming with auto-split and single-is_last discipline.

        Mirrors :meth:`SynthesisPipeline.synthesize_stream` but drives
        the :class:`BatchEngine` pool. Per-segment tail chunks are
        demoted to non-terminal; only the final segment's tail emits
        ``is_last=True`` with aggregated stats.
        """
        segments = self._segments(text)
        if not segments:
            raise ValueError("synthesize_stream called with empty text")

        collected_stats: list[SynthesisStats] = []
        for seg_idx, seg in enumerate(segments):
            is_final_segment = seg_idx == len(segments) - 1
            async for chunk in self.batch.synthesize_stream(seg, voice=voice, params=params):
                if chunk.is_last:
                    if chunk.stats is not None:
                        collected_stats.append(chunk.stats)
                    if is_final_segment:
                        yield SynthesisChunk(
                            audio=chunk.audio,
                            sample_rate=chunk.sample_rate,
                            is_last=True,
                            stats=_aggregate_stats(collected_stats),
                        )
                    else:
                        if len(chunk.audio) > 0:
                            yield SynthesisChunk(
                                audio=chunk.audio,
                                sample_rate=chunk.sample_rate,
                                is_last=False,
                                stats=None,
                            )
                else:
                    yield chunk
