"""FastAPI application for ``lunavox serve``.

Phase 5B update: the app now drives a :class:`BatchEngine` pool
under the hood, so the old ``asyncio.Lock`` that serialised 5A
requests is gone. Multiple clients hit :meth:`BatchEngine.submit`
and :meth:`BatchEngine.synthesize_stream` concurrently; the pool's
internal idle queue provides back-pressure when every engine is
busy. WebSocket streaming now supports every voice mode (base /
clone / custom / design).

Exposed endpoints:

* ``POST /v1/synth`` — one-shot synthesis, all voice modes, WAV
  body + stats header.
* ``WS /v1/stream`` — streaming WebSocket, all voice modes. Binary
  PCM chunks followed by a terminal JSON frame.
* ``GET /health``, ``GET /v1/models``.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
except ImportError as err:  # pragma: no cover — gated by [serve] extra
    raise ImportError('fastapi is required: pip install "lunavox[serve]"') from err

from lunavox.model import all_models
from lunavox.runtime import Voice

from .engine_holder import EngineHolder
from .schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SynthRequest,
    SynthResponseMeta,
    SynthStatsResponse,
)


def _build_voice(req: SynthRequest) -> Voice:
    """Translate a validated :class:`SynthRequest` into a :class:`Voice`."""
    if req.voice == "base":
        return Voice.base()
    if req.voice == "clone":
        if not req.reference:
            raise HTTPException(400, "voice=clone requires 'reference' path")
        return Voice.clone_file(req.reference)
    if req.voice == "custom":
        if not req.speaker:
            raise HTTPException(400, "voice=custom requires 'speaker'")
        return Voice.custom(req.speaker, instruct=req.instruct or "")
    if req.voice == "design":
        if not req.instruct:
            raise HTTPException(400, "voice=design requires 'instruct'")
        return Voice.design(req.instruct)
    raise HTTPException(400, f"Unknown voice mode: {req.voice}")


def _pcm_to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """Serialize a float32 mono array to an in-memory 16-bit PCM WAV."""
    pcm16 = bytearray()
    for sample in audio:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm16 += struct.pack("<h", int(clipped * 32767.0))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(bytes(pcm16))
    return buf.getvalue()


def _f32_to_pcm16_bytes(audio: Any) -> bytes:
    """Convert float32 samples in [-1, 1] to raw int16 LE PCM bytes."""
    out = bytearray()
    for sample in audio:
        clipped = max(-1.0, min(1.0, float(sample)))
        out += struct.pack("<h", int(clipped * 32767.0))
    return bytes(out)


def create_app(
    model_dir: Path,
    *,
    n_threads: int = 4,
    batch_size: int = 4,
) -> FastAPI:
    """Build the FastAPI app bound to one :class:`EngineHolder`.

    ``batch_size`` controls how many concurrent requests the pool
    can service — the Phase 5B default is 4, matching the CLI
    ``--batch-size`` flag. Pass ``batch_size=1`` for low-VRAM
    deployments; the class falls back to single-engine behaviour
    transparently.
    """
    holder = EngineHolder(
        model_dir=model_dir,
        batch_size=batch_size,
        n_threads=n_threads,
    )

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
        voice = _build_voice(req)
        params = holder.build_params(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_audio_tokens=req.max_audio_tokens,
        )

        # BatchEngine.submit acquires one idle engine from the pool,
        # runs synthesis on a background thread, and releases the
        # engine on completion. Concurrent callers queue on the pool's
        # internal asyncio.Queue, no explicit lock needed.
        result = await holder.batch.submit(req.text, voice=voice, params=params)

        wav_bytes = _pcm_to_wav_bytes(result.audio, result.sample_rate)
        meta = SynthResponseMeta(
            sample_rate=result.sample_rate,
            n_samples=int(len(result.audio)),
            mode=req.voice,
            stats=SynthStatsResponse(
                t_total_ms=result.stats.t_total_ms,
                audio_duration_ms=result.stats.audio_duration_ms,
                rtf=result.stats.rtf,
                rss_peak_bytes=result.stats.rss_peak_bytes,
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
          1. Client sends one text JSON frame matching :class:`SynthRequest`.
          2. Server sends binary frames containing raw int16 little-endian
             PCM chunks at the engine's sample rate (typically 24000).
          3. Server sends one terminal JSON frame with ``done=true`` and
             the :class:`SynthStatsResponse` payload, then closes.
        """
        await websocket.accept()

        try:
            raw = await websocket.receive_text()
            req = SynthRequest.model_validate_json(raw)
        except Exception as err:
            await websocket.close(code=1003, reason=f"invalid request: {err}")
            return

        try:
            voice = _build_voice(req)
        except HTTPException as err:
            await websocket.close(code=1003, reason=str(err.detail))
            return

        params = holder.build_params(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_audio_tokens=req.max_audio_tokens,
        )

        final_stats: Optional[SynthStatsResponse] = None
        try:
            async for chunk in holder.batch.synthesize_stream(req.text, voice=voice, params=params):
                if len(chunk.audio) > 0:
                    await websocket.send_bytes(_f32_to_pcm16_bytes(chunk.audio))
                if chunk.is_last and chunk.stats is not None:
                    final_stats = SynthStatsResponse(
                        t_total_ms=chunk.stats.t_total_ms,
                        audio_duration_ms=chunk.stats.audio_duration_ms,
                        rtf=chunk.stats.rtf,
                        rss_peak_bytes=chunk.stats.rss_peak_bytes,
                    )

            terminal = {
                "done": True,
                "sample_rate": holder.sample_rate,
                "stats": final_stats.model_dump() if final_stats else None,
            }
            await websocket.send_text(json.dumps(terminal))
        except WebSocketDisconnect:
            return
        except Exception as err:
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps({"error": str(err)}))
        finally:
            # Closing an already-closed socket raises; suppress quietly.
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


__all__ = ["create_app"]
