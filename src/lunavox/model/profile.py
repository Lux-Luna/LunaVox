"""Python dataclass view of ``models/<name>/model_profile.json``.

The same JSON is the source of truth for:

- ``src/lunavox_engine.cpp``'s C++ loader (via ``json_utils.h``
  extractors feeding ``lunavox::ModelProfile`` in
  ``src/model_profile.h``);
- this ``ModelProfile`` dataclass for Python consumers (GUI, tests,
  scripts);
- the CLI's display of model metadata.

Keep the two views aligned: every field added to ``ModelProfile`` here
must also exist in ``src/model_profile.h``, and vice versa.
See ``docs/en/technical/model_profile_schema.md`` for the shared
contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelProfile:
    """Subset of the JSON fields the Python side actually needs.

    Fields the GUI and scripts currently don't consume are intentionally
    not mirrored — they stay in the raw JSON under :attr:`raw` so
    nothing is lost, and the dataclass surface remains small enough to
    be stable.
    """

    version: int = 1
    model_type: str = "base"
    model_size: str = "unknown"
    instruct_support: bool = False

    talker_n_ctx: int = 0
    talker_n_ctx_train: int = 0
    predictor_n_ctx: int = 0

    codec_num_codebooks: int = 0
    codec_id_start: int = 0
    codec_id_end: int = 0

    default_max_new_tokens: int = 400
    default_temperature: float = 0.6
    default_top_p: float = 1.0
    default_top_k: int = 50
    default_repetition_penalty: float = 1.05

    # Raw passthrough of everything else in the JSON. Consumers that
    # need a field not yet mirrored above can read it from here without
    # waiting on a schema bump.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        size = self.model_size.upper() if self.model_size else "UNKNOWN"
        mtype = self.model_type.capitalize() if self.model_type else "Unknown"
        return f"Qwen3-TTS-12Hz-{size}-{mtype}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        # Use .get() with explicit defaults so a partial JSON (older
        # model variant, converted before a field was added) still
        # loads. The C++ side validates in is_valid() — this Python
        # view is intentionally permissive because most Python-side
        # consumers only care about display metadata.
        return cls(
            version=int(data.get("version", 1) or 1),
            model_type=str(data.get("model_type", "base") or "base"),
            model_size=str(data.get("model_size", "unknown") or "unknown"),
            instruct_support=bool(data.get("instruct_support", False)),
            talker_n_ctx=int(data.get("talker_n_ctx", 0) or 0),
            talker_n_ctx_train=int(data.get("talker_n_ctx_train", 0) or 0),
            predictor_n_ctx=int(data.get("predictor_n_ctx", 0) or 0),
            codec_num_codebooks=int(data.get("codec_num_codebooks", 0) or 0),
            codec_id_start=int(data.get("codec_id_start", 0) or 0),
            codec_id_end=int(data.get("codec_id_end", 0) or 0),
            default_max_new_tokens=int(data.get("default_max_new_tokens", 400) or 400),
            default_temperature=float(data.get("default_temperature", 0.6) or 0.6),
            default_top_p=float(data.get("default_top_p", 1.0) or 1.0),
            default_top_k=int(data.get("default_top_k", 50) or 50),
            default_repetition_penalty=float(data.get("default_repetition_penalty", 1.05) or 1.05),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ModelProfile":
        """Load ``model_profile.json`` from a model directory or file."""
        p = Path(path)
        if p.is_dir():
            p = p / "model_profile.json"
        with open(p, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"model_profile.json must be an object, got {type(data).__name__}")
        return cls.from_dict(data)
