"""Unified synthesis middle layer.

This package is the **single** place above :mod:`lunavox.runtime` that
CLI / HTTP / WebSocket / GUI code paths go through. The plumbing they
used to each re-implement — voice spec parsing, sampler default
merging, PCM → WAV encoding, long-text auto-splitting — lives here as
one implementation.

Layering:

* :mod:`lunavox.runtime` — C++ bindings. Knows nothing about voices as
  user intent, only about the four C ABI symbols.
* :mod:`lunavox.core.synth` — *this package*. Knows what a "voice
  request" means, how to turn it into an engine call, how to split
  long text, how to package audio. Re-exports the types callers need.
* :mod:`lunavox.cli` / :mod:`lunavox.serve` / :mod:`lunavox.gui` —
  thin entry adapters. Parse their native input (typer args / pydantic
  bodies / Tk variables), hand a :class:`VoiceSpec` + overrides to
  this layer, and render the result in their native output format.

Adding a new entry (e.g. gRPC) only means writing an adapter: the
synthesis logic itself does not need to be reimplemented.
"""

from __future__ import annotations

from .audio import f32_to_pcm16, f32_to_wav, pcm16_to_wav
from .pipeline import AsyncSynthesisPipeline, SynthesisPipeline
from .voice_spec import VoiceResolutionError, VoiceSpec, resolve_voice

__all__ = [
    "AsyncSynthesisPipeline",
    "SynthesisPipeline",
    "VoiceResolutionError",
    "VoiceSpec",
    "f32_to_pcm16",
    "f32_to_wav",
    "pcm16_to_wav",
    "resolve_voice",
]
