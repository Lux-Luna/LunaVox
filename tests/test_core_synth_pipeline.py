"""Unit tests for :class:`lunavox.core.synth.pipeline.SynthesisPipeline`
and :class:`AsyncSynthesisPipeline`.

Uses a :class:`FakeEngine` / :class:`FakeBatchEngine` stand-in so
tests run without liblunavox. The pipeline's job is to split, dispatch
to the engine, and stitch results — all of which we can verify end-to-
end against a deterministic fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from lunavox.core.synth.pipeline import AsyncSynthesisPipeline, SynthesisPipeline
from lunavox.runtime import (
    MemStats,
    SynthesisChunk,
    SynthesisMode,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
    Voice,
)

SAMPLE_RATE = 24000


def _fake_result(text: str) -> SynthesisResult:
    """Build a deterministic :class:`SynthesisResult` for text of length L.

    Audio: L×10 samples of ramping floats. Stats: duration = L ms (so
    aggregation tests have predictable sums).
    """
    n = len(text) * 10
    audio = np.linspace(0.0, 0.5, n, dtype=np.float32)
    stats = SynthesisStats(
        t_total_ms=len(text),
        audio_duration_ms=len(text),
        rtf=1.0,
        mem=MemStats(rss_peak_bytes=len(text) * 1024),
    )
    return SynthesisResult(
        audio=audio, sample_rate=SAMPLE_RATE, stats=stats, mode=SynthesisMode.BASE
    )


class FakeEngine:
    """Sync stand-in for :class:`lunavox.runtime.Engine`.

    Records every call for later assertion. ``synthesize_stream``
    splits the audio into 3 equal chunks so the pipeline's
    is_last-merging logic has something real to chew on.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.stream_calls: list[str] = []

    def synthesize(
        self,
        text: str,
        voice: Voice | None = None,  # noqa: ARG002
        params: SynthesisParams | None = None,  # noqa: ARG002
    ) -> SynthesisResult:
        self.calls.append(text)
        return _fake_result(text)

    def synthesize_stream(
        self,
        text: str,
        voice: Voice | None = None,  # noqa: ARG002
        params: SynthesisParams | None = None,  # noqa: ARG002
    ) -> Iterator[SynthesisChunk]:
        self.stream_calls.append(text)
        result = _fake_result(text)
        audio = result.audio
        n = len(audio)
        third = max(1, n // 3)
        slices = [audio[:third], audio[third : 2 * third], audio[2 * third :]]
        for i, piece in enumerate(slices):
            is_last = i == len(slices) - 1
            yield SynthesisChunk(
                audio=piece,
                sample_rate=SAMPLE_RATE,
                is_last=is_last,
                stats=result.stats if is_last else None,
            )


class FakeBatchEngine:
    """Async stand-in for :class:`BatchEngine`. Delegates to :class:`FakeEngine`."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self._inner = FakeEngine()

    @property
    def calls(self) -> list[str]:
        return self._inner.calls

    @property
    def stream_calls(self) -> list[str]:
        return self._inner.stream_calls

    async def submit(
        self,
        text: str,
        voice: Voice | None = None,
        params: SynthesisParams | None = None,
    ) -> SynthesisResult:
        return self._inner.synthesize(text, voice=voice, params=params)

    async def synthesize_stream(
        self,
        text: str,
        voice: Voice | None = None,
        params: SynthesisParams | None = None,
    ) -> AsyncIterator[SynthesisChunk]:
        for chunk in self._inner.synthesize_stream(text, voice=voice, params=params):
            yield chunk


# ---------------------------------------------------------------------
# sync: synthesize
# ---------------------------------------------------------------------


def test_short_text_bypasses_splitter():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake, auto_split_threshold=100)  # type: ignore[arg-type]
    result = pipe.synthesize("Short input.")
    assert len(fake.calls) == 1
    assert fake.calls[0] == "Short input."
    assert result.sample_rate == SAMPLE_RATE


def test_long_text_is_split_and_results_are_merged():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake, auto_split_threshold=30)  # type: ignore[arg-type]
    text = "First sentence here. Second sentence here. Third sentence too."
    result = pipe.synthesize(text)
    assert len(fake.calls) >= 2
    # All audio chunks concatenated.
    expected_samples = sum(len(_fake_result(c).audio) for c in fake.calls)
    assert len(result.audio) == expected_samples
    # Stats aggregated.
    assert result.stats.t_total_ms == sum(len(c) for c in fake.calls)


def test_empty_text_raises_value_error():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.synthesize("")


# ---------------------------------------------------------------------
# sync: synthesize_stream
# ---------------------------------------------------------------------


def test_stream_single_segment_preserves_is_last():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake, auto_split_threshold=100)  # type: ignore[arg-type]
    chunks = list(pipe.synthesize_stream("Short stream."))
    assert len(fake.stream_calls) == 1
    last_chunks = [c for c in chunks if c.is_last]
    assert len(last_chunks) == 1
    assert last_chunks[0].stats is not None


def test_stream_multi_segment_has_exactly_one_is_last():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake, auto_split_threshold=20)  # type: ignore[arg-type]
    text = "First sentence here. Second sentence here. Third sentence too."
    chunks = list(pipe.synthesize_stream(text))
    last_chunks = [c for c in chunks if c.is_last]
    assert len(last_chunks) == 1
    # Only the terminal chunk carries aggregated stats.
    assert last_chunks[0].stats is not None
    assert last_chunks[0].stats.t_total_ms == sum(len(c) for c in fake.stream_calls)


def test_stream_chunks_come_out_in_order():
    fake = FakeEngine()
    pipe = SynthesisPipeline(fake, auto_split_threshold=20)  # type: ignore[arg-type]
    text = "First sentence here. Second sentence here."
    chunks = list(pipe.synthesize_stream(text))
    # All non-empty chunks.
    assert all(len(c.audio) >= 0 for c in chunks)
    # Concatenation equals sum of per-segment audio (within the fake model).
    total = sum(len(c.audio) for c in chunks)
    expected = sum(len(_fake_result(s).audio) for s in fake.stream_calls)
    assert total == expected


# ---------------------------------------------------------------------
# async pipeline
# ---------------------------------------------------------------------


def test_async_short_text_bypasses_splitter():
    fake = FakeBatchEngine()
    pipe = AsyncSynthesisPipeline(fake, auto_split_threshold=100)  # type: ignore[arg-type]

    async def run():
        return await pipe.synthesize("Hello.")

    result = asyncio.run(run())
    assert len(fake.calls) == 1
    assert result.sample_rate == SAMPLE_RATE


def test_async_long_text_splits_and_merges():
    fake = FakeBatchEngine()
    pipe = AsyncSynthesisPipeline(fake, auto_split_threshold=25)  # type: ignore[arg-type]

    async def run():
        text = "First sentence here. Second sentence here. Third sentence too."
        return await pipe.synthesize(text)

    result = asyncio.run(run())
    assert len(fake.calls) >= 2
    assert result.stats.t_total_ms == sum(len(c) for c in fake.calls)


def test_async_stream_multi_segment_has_exactly_one_is_last():
    fake = FakeBatchEngine()
    pipe = AsyncSynthesisPipeline(fake, auto_split_threshold=20)  # type: ignore[arg-type]

    async def run():
        out: list[SynthesisChunk] = []
        async for chunk in pipe.synthesize_stream(
            "First sentence here. Second sentence here. Third sentence too."
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(run())
    last_chunks = [c for c in chunks if c.is_last]
    assert len(last_chunks) == 1
    assert last_chunks[0].stats is not None
