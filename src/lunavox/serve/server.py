"""FastAPI application for ``lunavox serve``.

Exposes:

* ``POST /v1/synth`` — one-shot synthesis; accepts all four voice modes
  (base, clone, custom, design), returns a WAV body with metadata in
  the ``X-Lunavox-Stats`` header.
* ``WS /v1/stream`` — WebSocket sentence-streaming. Client sends a
  :class:`SynthRequest` JSON frame, the server pushes binary PCM
  chunks as they become available, and closes with a terminal JSON
  frame carrying :class:`SynthStatsResponse` stats. Base voice only
  in Phase 5A (see Engine.synthesize_stream).
* ``GET /health``, ``GET /v1/models`` — standard liveness / catalog.

Concurrency: a single :class:`EngineHolder` owned by the app state
holds an ``asyncio.Lock``. Every request acquires the lock before
entering the C engine, so concurrent clients queue politely on one
GPU. Phase 5B will swap this for a real BatchEngine without changing
the handler shape.
"""

from __future__ import annotations

import asyncio
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


def create_app(model_dir: Path, n_threads: int = 4) -> FastAPI:
    """Build the FastAPI app bound to one :class:`EngineHolder`.

    Kept as a factory so tests (and future multi-model deployments)
    can instantiate isolated apps without module-level globals.
    """
    holder = EngineHolder(model_dir=model_dir, n_threads=n_threads)

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
        if holder._engine is None:
            return HealthResponse(status="loading", model=None, sample_rate=None)
        return HealthResponse(
            status="ok",
            model=holder.model_dir.name,
            sample_rate=holder.engine.sample_rate,
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

        async with holder.lock:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: holder.engine.synthesize(req.text, voice=voice, params=params),
            )

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
        """Sentence-streaming endpoint.

        Protocol:
          1. Client sends one text JSON frame matching :class:`SynthRequest`
             (voice must be ``base`` in Phase 5A — other modes close 1003).
          2. Server sends binary frames containing raw int16 little-endian
             PCM chunks at the engine's sample rate (typically 24000).
          3. Server sends one terminal JSON frame with key ``done=true``
             and the :class:`SynthStatsResponse` payload, then closes.
        """
        await websocket.accept()

        try:
            raw = await websocket.receive_text()
            req = SynthRequest.model_validate_json(raw)
        except Exception as err:
            await websocket.close(code=1003, reason=f"invalid request: {err}")
            return

        if req.voice != "base":
            await websocket.close(
                code=1003,
                reason=f"WebSocket streaming is base-only in Phase 5A (got {req.voice})",
            )
            return

        params = holder.build_params(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_audio_tokens=req.max_audio_tokens,
        )

        try:
            async with holder.lock:
                loop = asyncio.get_running_loop()
                chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
                sentinel: dict[str, Any] = {"done": False}

                def _producer() -> None:
                    try:
                        for chunk in holder.engine.synthesize_stream(
                            req.text, voice=Voice.base(), params=params
                        ):
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                    except BaseException as err:
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait,
                            {"error": str(err)},
                        )
                    finally:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)

                producer = loop.run_in_executor(None, _producer)
                final_stats: Optional[SynthStatsResponse] = None

                while True:
                    item = await chunk_queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, dict) and "error" in item:
                        await websocket.send_text(json.dumps({"error": item["error"]}))
                        break
                    # Convert float32 to int16 PCM and push as binary frame.
                    int16 = bytearray()
                    for sample in item.audio:
                        clipped = max(-1.0, min(1.0, float(sample)))
                        int16 += struct.pack("<h", int(clipped * 32767.0))
                    if int16:
                        await websocket.send_bytes(bytes(int16))
                    if item.is_last and item.stats is not None:
                        final_stats = SynthStatsResponse(
                            t_total_ms=item.stats.t_total_ms,
                            audio_duration_ms=item.stats.audio_duration_ms,
                            rtf=item.stats.rtf,
                            rss_peak_bytes=item.stats.rss_peak_bytes,
                        )

                await producer  # reap worker

                terminal = {
                    "done": True,
                    "sample_rate": holder.engine.sample_rate,
                    "stats": final_stats.model_dump() if final_stats else None,
                }
                await websocket.send_text(json.dumps(terminal))
        except WebSocketDisconnect:
            return
        finally:
            # Closing an already-closed socket raises; suppress quietly.
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


__all__ = ["create_app"]
