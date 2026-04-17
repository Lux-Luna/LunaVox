"""Single source of truth for turning a voice request into a :class:`Voice`.

Before this module, every entry — CLI ``--voice`` flags, HTTP
``SynthRequest``, GUI ``VoicePicker`` — re-implemented the same
``if voice == "base" ... elif voice == "clone" ...`` ladder against
slightly different input types and raised different exception styles.
Any new voice mode had to be added in four places.

Now:

1. Each entry parses its native inputs into a :class:`VoiceSpec`.
2. :func:`resolve_voice` is the **only** implementation of the
   ``VoiceSpec → Voice`` mapping.
3. Validation errors raise :class:`VoiceResolutionError`, which the
   entry adapts to its own UX (HTTP 400 / typer BadParameter / GUI
   dialog / whatever).

Adding a voice mode is a two-file change: extend
:class:`lunavox.runtime.voice.Voice` with a factory, add one branch
to :func:`resolve_voice`. No entry-adapter changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from lunavox.runtime import Voice

VoiceMode = Literal["base", "clone", "custom", "design"]


class VoiceResolutionError(ValueError):
    """Raised when a :class:`VoiceSpec` is missing fields required by its mode.

    Subclass of :class:`ValueError` so callers that already handle the
    stdlib hierarchy get it for free. Entries that need to translate
    it to their own error shape (HTTP 400, typer exit) should catch
    this specific type.
    """


@dataclass(frozen=True)
class VoiceSpec:
    """Mode-parametric, transport-agnostic description of how to voice a request.

    Each entry builds one of these from its native inputs (typer
    options, pydantic model, Tk widget state) so the downstream
    :func:`resolve_voice` call doesn't know or care which frontend
    originated the request.

    Fields map 1:1 onto the HTTP ``SynthRequest`` schema so the
    round-trip is trivial.
    """

    mode: VoiceMode = "base"
    reference: Optional[str] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None

    @classmethod
    def base(cls) -> VoiceSpec:
        return cls(mode="base")

    @classmethod
    def clone(cls, reference: str) -> VoiceSpec:
        return cls(mode="clone", reference=reference)

    @classmethod
    def custom(cls, speaker: str, instruct: Optional[str] = None) -> VoiceSpec:
        return cls(mode="custom", speaker=speaker, instruct=instruct)

    @classmethod
    def design(cls, instruct: str) -> VoiceSpec:
        return cls(mode="design", instruct=instruct)


def resolve_voice(spec: VoiceSpec) -> Voice:
    """Convert a :class:`VoiceSpec` into a runtime :class:`Voice`.

    This is the **only** canonical implementation of that mapping —
    CLI / HTTP / GUI all call in here. Raises
    :class:`VoiceResolutionError` on missing required fields; the
    entry adapts the exception to its own user-facing error shape.
    """
    if spec.mode == "base":
        return Voice.base()
    if spec.mode == "clone":
        if not spec.reference:
            raise VoiceResolutionError("voice=clone requires 'reference' (path to .wav or .json)")
        return Voice.clone_file(Path(spec.reference))
    if spec.mode == "custom":
        if not spec.speaker:
            raise VoiceResolutionError("voice=custom requires 'speaker' (catalog id)")
        return Voice.custom(spec.speaker, instruct=spec.instruct or "")
    if spec.mode == "design":
        if not spec.instruct or not spec.instruct.strip():
            raise VoiceResolutionError("voice=design requires non-empty 'instruct' text")
        return Voice.design(spec.instruct)
    raise VoiceResolutionError(f"Unknown voice mode: {spec.mode!r}")
