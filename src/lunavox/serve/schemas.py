"""Pydantic request/response models for the HTTP API.

Kept in a dedicated module so tests can validate the schema surface
without importing FastAPI or starting a server. Every field here is
user-facing JSON; changes are ABI-breaking for clients.
"""

from __future__ import annotations

from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field
except ImportError as err:  # pragma: no cover — gated by [serve] extra
    raise ImportError('fastapi/pydantic are required: pip install "lunavox[serve]"') from err


VoiceMode = Literal["base", "clone", "custom", "design"]


class SynthRequest(BaseModel):
    """Payload for ``POST /v1/synth`` and ``WS /v1/stream``."""

    text: str = Field(min_length=1, description="Text to synthesize.")
    voice: VoiceMode = Field(
        default="base",
        description="Voice mode. WebSocket streaming supports base only in Phase 5A.",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Path to a .wav or .json reference file when voice=clone.",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Catalog speaker id when voice=custom.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Style / design instruction for voice=custom or voice=design.",
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=1000)
    max_audio_tokens: Optional[int] = Field(default=None, ge=0)


class SynthStatsResponse(BaseModel):
    """Per-run stats echoed on every synthesis response."""

    t_total_ms: int
    audio_duration_ms: int
    rtf: float
    rss_peak_bytes: int


class SynthResponseMeta(BaseModel):
    """JSON envelope returned by ``POST /v1/synth`` alongside the WAV body.

    When ``Accept: application/json`` (the default), the server returns
    this envelope with a base64-encoded ``wav`` field. When ``Accept:
    audio/wav``, the raw WAV bytes come back directly and this envelope
    is returned only in the ``X-Lunavox-Stats`` header as a compact
    JSON string.
    """

    sample_rate: int
    n_samples: int
    mode: VoiceMode
    stats: SynthStatsResponse


class ModelInfo(BaseModel):
    name: str
    display_name: str
    repo_id: str
    installed: bool


class ModelsResponse(BaseModel):
    active: Optional[str]
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: Optional[str]
    sample_rate: Optional[int]
    detail: Optional[str] = None


class TextStreamInit(BaseModel):
    """First JSON frame on ``WS /v1/stream/text``.

    Carries the voice + sampler config; the actual text comes
    later as a sequence of ``TextStreamChunk`` frames followed by
    a ``TextStreamEnd`` frame. Mirrors the ``SynthRequest`` field
    set, minus ``text`` itself.
    """

    voice: VoiceMode = Field(default="base")
    reference: Optional[str] = Field(default=None)
    speaker: Optional[str] = Field(default=None)
    instruct: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=1000)
    max_audio_tokens: Optional[int] = Field(default=None, ge=0)


class TextStreamChunk(BaseModel):
    """A chunk of incoming text on the streaming-input WS channel."""

    text: str = Field(min_length=1)


class TextStreamEnd(BaseModel):
    """Sentinel frame that closes the streaming-input channel.

    The server flushes any buffered partial sentence as the final
    synthesis unit, then closes the WebSocket cleanly.
    """

    end: Literal[True]
