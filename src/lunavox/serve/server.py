"""FastAPI application for ``lunavox serve``.

Refactored to go through :mod:`lunavox.core.synth` for every call.
Endpoint handlers no longer know about:

* **voice resolution** — requests call their own ``to_voice_spec()``
  and hand it to :func:`resolve_voice`, one implementation in core.
* **param building** — ``SynthesisParams.from_overrides(**req.param_overrides())``,
  single default source.
* **PCM encoding** — :func:`f32_to_pcm16` / :func:`pcm16_to_wav`
  vectorized numpy, ~100x faster than the per-sample loop they
  replaced.
* **long-text splitting** — :class:`AsyncSynthesisPipeline` auto-
  splits anything over its threshold before dispatching to the
  engine pool. Any endpoint's input text is transparently handled.

Endpoints expose the same wire protocol as before:

* ``POST /v1/synth`` — one-shot, WAV body + ``X-Lunavox-Stats`` header.
* ``WS /v1/stream`` — binary PCM frames + terminal JSON.
* ``WS /v1/stream/text`` — LLM-streaming input, same output shape,
  plus ``sentences`` count in the terminal frame.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect

from lunavox.core.synth import (
    VoiceResolutionError,
    f32_to_pcm16,
    pcm16_to_wav,
    resolve_voice,
)
from lunavox.core.text import StreamingSentenceBuffer
from lunavox.model import all_models
from lunavox.runtime import SynthesisParams, Voice

from .engine_holder import EngineHolder
from .metrics import LunavoxMetrics
from .schemas import (
    HealthResponse,
    MemStatsResponse,
    ModelInfo,
    ModelsResponse,
    SynthRequest,
    SynthResponseMeta,
    SynthStatsResponse,
    TextStreamChunk,
    TextStreamEnd,
    TextStreamInit,
)


def _mem_response(stats: Any) -> MemStatsResponse:
    """Project a :class:`SynthesisStats.mem` sub-struct onto the wire model.

    Folding this into a helper keeps every `SynthStatsResponse(...)`
    construction site one line shorter and ensures all three endpoints
    agree on the field list — the HTTP one-shot, the WS audio stream,
    and the WS text stream used to drift independently.
    """
    mem = stats.mem
    return MemStatsResponse(
        rss_start_bytes=mem.rss_start_bytes,
        rss_end_bytes=mem.rss_end_bytes,
        rss_peak_bytes=mem.rss_peak_bytes,
        vram_start_bytes=mem.vram_start_bytes,
        vram_end_bytes=mem.vram_end_bytes,
        vram_peak_bytes=mem.vram_peak_bytes,
        vram_measured=mem.vram_measured,
    )


def _resolve_voice_or_400(req: SynthRequest | TextStreamInit) -> Voice:
    """Bridge :class:`VoiceResolutionError` to an HTTP 400 response."""
    try:
        return resolve_voice(req.to_voice_spec())
    except VoiceResolutionError as err:
        raise HTTPException(400, str(err)) from err


def create_app(
    model_dir: Path,
    *,
    n_threads: int = 4,
    batch_size: int = 4,
    auto_split_threshold: int = 240,
) -> FastAPI:
    """Build the FastAPI app bound to one :class:`EngineHolder`.

    ``auto_split_threshold`` is the char count above which incoming
    text is automatically chunked by the pipeline — see
    :class:`~lunavox.core.synth.AsyncSynthesisPipeline`. Default 240
    is tuned for real-time voice agent responses; set higher for
    batch / offline jobs where you want fewer segment boundaries.
    """
    holder = EngineHolder(
        model_dir=model_dir,
        batch_size=batch_size,
        n_threads=n_threads,
        auto_split_threshold=auto_split_threshold,
    )
    metrics = LunavoxMetrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await holder.load()
        try:
            yield
        finally:
            holder.close()

    app = FastAPI(
        title="LunaVox",
        version="2.2.0",
        description="HTTP / WebSocket serving layer for LunaVox Qwen3-TTS.",
        lifespan=lifespan,
    )
    app.state.holder = holder
    app.state.metrics = metrics

    # ---- health / models ---------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if holder._batch is None:
            return HealthResponse(status="loading", model=None, sample_rate=None)
        return HealthResponse(
            status="ok",
            model=holder.model_dir.name,
            sample_rate=holder.sample_rate,
        )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        batch = holder._batch
        body, content_type = metrics.render(batch)
        return Response(content=body, media_type=content_type)

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models_list() -> ModelsResponse:
        models = []
        project_models = holder.model_dir.parent
        for spec in all_models():
            local = project_models / spec.name
            installed = local.exists() and any(local.iterdir())
            models.append(
                ModelInfo(
                    name=spec.name,
                    display_name=spec.display_name,
                    repo_id=spec.repo_id,
                    installed=installed,
                )
            )
        return ModelsResponse(active=holder.model_dir.name, models=models)

    # ---- one-shot synthesis ------------------------------------------

    @app.post("/v1/synth")
    async def synth(req: SynthRequest) -> Response:
        voice = _resolve_voice_or_400(req)
        params = SynthesisParams.from_overrides(**req.param_overrides())
        params.n_threads = holder.n_threads  # stays configured by the server

        metrics.snapshot_pool(holder.batch)
        t_start = time.perf_counter()
        try:
            result = await holder.pipeline.synthesize(req.text, voice=voice, params=params)
        except Exception:
            metrics.requests_total.labels(voice=req.voice, status="error").inc()
            raise
        elapsed = time.perf_counter() - t_start
        metrics.requests_total.labels(voice=req.voice, status="success").inc()
        metrics.request_duration_seconds.labels(voice=req.voice).observe(elapsed)
        if result.stats.rtf > 0:
            metrics.rtf.labels(voice=req.voice).observe(result.stats.rtf)

        wav_bytes = pcm16_to_wav(f32_to_pcm16(result.audio), result.sample_rate)
        meta = SynthResponseMeta(
            sample_rate=result.sample_rate,
            n_samples=int(len(result.audio)),
            mode=req.voice,
            stats=SynthStatsResponse(
                t_total_ms=result.stats.t_total_ms,
                audio_duration_ms=result.stats.audio_duration_ms,
                rtf=result.stats.rtf,
                mem=_mem_response(result.stats),
            ),
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Lunavox-Stats": meta.model_dump_json()},
        )

    # ---- streaming WebSocket -----------------------------------------

    @app.websocket("/v1/stream")
    async def stream_ws(websocket: WebSocket) -> None:
        """Sentence-streaming endpoint — all four voice modes.

        Protocol:
          1. Client sends one JSON :class:`SynthRequest` text frame.
          2. Server sends binary int16 LE PCM frames.
          3. Server sends one terminal JSON frame with ``done=true``
             and the :class:`SynthStatsResponse` payload, then closes.

        Long inputs are auto-split by the pipeline — client sees one
        continuous PCM stream regardless of how many segments the
        engine synthesized under the hood.
        """
        await websocket.accept()

        try:
            raw = await websocket.receive_text()
            req = SynthRequest.model_validate_json(raw)
        except Exception as err:
            await websocket.close(code=1003, reason=f"invalid request: {err}")
            return

        try:
            voice = resolve_voice(req.to_voice_spec())
        except VoiceResolutionError as err:
            await websocket.close(code=1003, reason=str(err))
            return

        params = SynthesisParams.from_overrides(**req.param_overrides())
        params.n_threads = holder.n_threads

        metrics.snapshot_pool(holder.batch)
        t_start = time.perf_counter()
        final_stats: Optional[SynthStatsResponse] = None
        success = False
        try:
            async for chunk in holder.pipeline.synthesize_stream(
                req.text, voice=voice, params=params
            ):
                if len(chunk.audio) > 0:
                    await websocket.send_bytes(f32_to_pcm16(chunk.audio))
                if chunk.is_last and chunk.stats is not None:
                    final_stats = SynthStatsResponse(
                        t_total_ms=chunk.stats.t_total_ms,
                        audio_duration_ms=chunk.stats.audio_duration_ms,
                        rtf=chunk.stats.rtf,
                        mem=_mem_response(chunk.stats),
                    )

            terminal = {
                "done": True,
                "sample_rate": holder.sample_rate,
                "stats": final_stats.model_dump() if final_stats else None,
            }
            await websocket.send_text(json.dumps(terminal))
            success = True
        except WebSocketDisconnect:
            return
        except Exception as err:
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps({"error": str(err)}))
        finally:
            elapsed = time.perf_counter() - t_start
            status = "success" if success else "error"
            metrics.requests_total.labels(voice=req.voice, status=status).inc()
            metrics.request_duration_seconds.labels(voice=req.voice).observe(elapsed)
            if final_stats is not None and final_stats.rtf > 0:
                metrics.rtf.labels(voice=req.voice).observe(final_stats.rtf)
            with contextlib.suppress(Exception):
                await websocket.close()

    # ---- streaming-input WebSocket -----------------------------------

    @app.websocket("/v1/stream/text")
    async def stream_text_ws(websocket: WebSocket) -> None:
        """Sentence-streaming **input** endpoint for voice agents.

        Lets a caller (typically an LLM streaming reply tokens) push
        text into the synthesizer one chunk at a time and start
        receiving audio after each complete sentence.

        Protocol:
          1. Client sends one JSON :class:`TextStreamInit` frame.
          2. Client sends N JSON :class:`TextStreamChunk` frames.
          3. Server watches the buffer for sentence boundaries via
             :class:`StreamingSentenceBuffer`, dispatches each complete
             sentence through the pipeline (which itself may sub-split
             overlong sentences), and pushes back binary PCM frames.
          4. Client sends one :class:`TextStreamEnd` frame; server
             flushes any partial trailing text.
          5. Server sends one terminal JSON frame with ``done=true``,
             ``sentences`` count, and final stats.
        """
        await websocket.accept()

        try:
            raw = await websocket.receive_text()
            init = TextStreamInit.model_validate_json(raw)
        except Exception as err:
            await websocket.close(code=1003, reason=f"invalid init frame: {err}")
            return

        try:
            voice = resolve_voice(init.to_voice_spec())
        except VoiceResolutionError as err:
            await websocket.close(code=1003, reason=str(err))
            return

        params = SynthesisParams.from_overrides(**init.param_overrides())
        params.n_threads = holder.n_threads

        buffer = StreamingSentenceBuffer()
        sentences_synthed = 0
        last_stats: Optional[SynthStatsResponse] = None
        success = False
        t_start = time.perf_counter()

        async def _synth_sentence(sentence: str) -> None:
            nonlocal sentences_synthed, last_stats
            sentences_synthed += 1
            async for chunk in holder.pipeline.synthesize_stream(
                sentence, voice=voice, params=params
            ):
                if len(chunk.audio) > 0:
                    await websocket.send_bytes(f32_to_pcm16(chunk.audio))
                if chunk.is_last and chunk.stats is not None:
                    last_stats = SynthStatsResponse(
                        t_total_ms=chunk.stats.t_total_ms,
                        audio_duration_ms=chunk.stats.audio_duration_ms,
                        rtf=chunk.stats.rtf,
                        mem=_mem_response(chunk.stats),
                    )

        try:
            while True:
                raw_frame = await websocket.receive_text()
                try:
                    payload = json.loads(raw_frame)
                except ValueError as err:
                    await websocket.send_text(json.dumps({"error": f"invalid JSON frame: {err}"}))
                    continue

                if isinstance(payload, dict) and payload.get("end") is True:
                    TextStreamEnd.model_validate(payload)
                    for leftover in buffer.flush():
                        await _synth_sentence(leftover)
                    break

                try:
                    chunk_frame = TextStreamChunk.model_validate(payload)
                except Exception as err:
                    await websocket.send_text(json.dumps({"error": f"invalid chunk frame: {err}"}))
                    continue

                buffer.feed(chunk_frame.text)
                for sentence in buffer.drain():
                    await _synth_sentence(sentence)

            terminal = {
                "done": True,
                "sample_rate": holder.sample_rate,
                "sentences": sentences_synthed,
                "stats": last_stats.model_dump() if last_stats else None,
            }
            await websocket.send_text(json.dumps(terminal))
            success = True
        except WebSocketDisconnect:
            return
        except Exception as err:
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps({"error": str(err)}))
        finally:
            elapsed = time.perf_counter() - t_start
            status = "success" if success else "error"
            metrics.requests_total.labels(voice=init.voice, status=status).inc()
            metrics.request_duration_seconds.labels(voice=init.voice).observe(elapsed)
            if last_stats is not None and last_stats.rtf > 0:
                metrics.rtf.labels(voice=init.voice).observe(last_stats.rtf)
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


__all__ = ["create_app"]
