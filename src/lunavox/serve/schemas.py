"""Pydantic request/response models for the HTTP API.

Kept in a dedicated module so tests can validate the schema surface
without importing FastAPI or starting a server. Every field here is
user-facing JSON; changes are ABI-breaking for clients.

Request schemas implement :meth:`to_voice_spec` and
:meth:`param_overrides` so the FastAPI handlers never have to know
which fields belong to voice vs sampler — they just translate
pydantic → :mod:`lunavox.core.synth` inputs and call the pipeline.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from lunavox.core.synth import VoiceSpec

VoiceMode = Literal["base", "clone", "custom", "design"]


class _VoiceFields(BaseModel):
    """Shared voice / sampler fields reused by every request schema.

    Mixed into both ``SynthRequest`` (one-shot + ``WS /v1/stream``) and
    ``TextStreamInit`` (``WS /v1/stream/text``) so the ``VoiceSpec``
    and param-override conversions have one implementation.
    """

    voice: VoiceMode = Field(default="base")
    reference: Optional[str] = Field(default=None)
    speaker: Optional[str] = Field(default=None)
    instruct: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=1000)
    max_audio_tokens: Optional[int] = Field(default=None, ge=0)

    def to_voice_spec(self) -> VoiceSpec:
        """Build the transport-agnostic :class:`VoiceSpec` for this request."""
        return VoiceSpec(
            mode=self.voice,
            reference=self.reference,
            speaker=self.speaker,
            instruct=self.instruct,
        )

    def param_overrides(self) -> dict[str, Any]:
        """Collect the sampler overrides as a kwargs dict for
        :meth:`SynthesisParams.from_overrides`. ``None`` values are
        kept — the downstream helper filters them out so callers
        don't need per-field ``if`` guards."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_audio_tokens": self.max_audio_tokens,
        }


class SynthRequest(_VoiceFields):
    """Payload for ``POST /v1/synth`` and ``WS /v1/stream``.

    Voice/sampler fields come from :class:`_VoiceFields`; only the
    mandatory ``text`` field is added here.
    """

    text: str = Field(min_length=1, description="Text to synthesize.")


class MemStatsResponse(BaseModel):
    """Mirror of :class:`lunavox.runtime.MemStats`.

    All samples come from the same in-engine checkpoints, so peaks and
    starts are self-consistent. ``vram_measured`` is the authoritative
    "did NVML return real numbers" flag — clients MUST NOT rely on
    ``vram_peak_bytes > 0`` (a zero reading on a CPU-only run is a valid
    measurement, not "unavailable"). When ``vram_measured`` is false the
    ``vram_*`` fields are undefined.
    """

    rss_start_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int
    vram_start_bytes: int = 0
    vram_end_bytes: int = 0
    vram_peak_bytes: int = 0
    vram_measured: bool = False


class SynthStatsResponse(BaseModel):
    """Per-run stats echoed on every synthesis response."""

    t_total_ms: int
    audio_duration_ms: int
    rtf: float
    mem: MemStatsResponse


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


class TextStreamInit(_VoiceFields):
    """First JSON frame on ``WS /v1/stream/text``.

    Carries the voice + sampler config; the actual text comes
    later as a sequence of ``TextStreamChunk`` frames followed by
    a ``TextStreamEnd`` frame. Mirrors the :class:`SynthRequest`
    field set, minus ``text`` itself.
    """


class TextStreamChunk(BaseModel):
    """A chunk of incoming text on the streaming-input WS channel."""

    text: str = Field(min_length=1)


class TextStreamEnd(BaseModel):
    """Sentinel frame that closes the streaming-input channel.

    The server flushes any buffered partial sentence as the final
    synthesis unit, then closes the WebSocket cleanly.
    """

    end: Literal[True]
