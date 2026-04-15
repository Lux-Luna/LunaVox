"""The :class:`Voice` one-shot input object.

Before this refactor the engine exposed a 6-way family of
``synthesize_*`` methods that all took the same ``text`` + ``params``
pair plus mode-specific extras. Every new voice mode meant editing the
engine, the CLI, and every binding site. :class:`Voice` folds the
mode-specific bits into a single first-class argument so
``Engine.synthesize(text, voice, params)`` is the only entry point —
adding a new voice mode means adding a classmethod here and one branch
inside ``Engine._dispatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .params import SynthesisMode


@dataclass(frozen=True)
class Voice:
    """A description of how to voice a synthesis request.

    Do not construct directly — use the classmethod factories:

    * ``Voice.base()`` — default speaker baked into the model
    * ``Voice.clone_file(path)`` — reference WAV or pre-computed JSON features
    * ``Voice.custom(speaker, instruct=...)`` — catalog speaker id with optional style
    * ``Voice.design(instruct=...)`` — text-described synthetic voice
    """

    mode: SynthesisMode
    reference_path: Optional[Path] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None

    @classmethod
    def base(cls) -> Voice:
        return cls(mode=SynthesisMode.BASE)

    @classmethod
    def clone_file(cls, path: Union[str, Path]) -> Voice:
        """Clone a voice from a reference ``.wav`` or pre-computed ``.json``
        feature file. The C engine auto-detects the format from the extension."""
        p = Path(path)
        return cls(mode=SynthesisMode.CLONE_FILE, reference_path=p)

    @classmethod
    def custom(cls, speaker: str, instruct: str = "") -> Voice:
        """Use a catalog speaker id (e.g. ``"Vivian"``) with an optional
        free-form style instruction."""
        if not speaker:
            raise ValueError("Voice.custom requires a non-empty speaker id")
        return cls(mode=SynthesisMode.CUSTOM, speaker=speaker, instruct=instruct)

    @classmethod
    def design(cls, instruct: str) -> Voice:
        """Synthesise a speaker from a text description. The ``instruct``
        field is required — an empty string is rejected at the Engine
        level too, but catching it here gives a clearer error."""
        if not instruct.strip():
            raise ValueError("Voice.design requires a non-empty instruct text")
        return cls(mode=SynthesisMode.DESIGN, instruct=instruct)
