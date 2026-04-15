"""Single source of truth for the Qwen3-TTS model catalog.

Every other module (CLI prompts, downloader, conversion pipeline) reads
from ``MODELS`` defined here. Adding or renaming a model means touching
this file and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

HF_HUB_ROOT = Path(HF_HUB_CACHE)
HF_ORG = "Qwen"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Catalog entry for one published Qwen3-TTS model."""

    name: str            # internal key used by CLI flags and models/<name>/
    repo: str            # HuggingFace repo basename (under HF_ORG)
    display_name: str    # human-readable label for prompts / logs
    size: str            # "0.6B" / "1.7B"
    mode: str            # "base" / "custom" / "design"

    @property
    def repo_id(self) -> str:
        return f"{HF_ORG}/{self.repo}"


# Registry order determines the order shown in interactive prompts and
# `--help` enumerations — list smallest / most common first.
_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("base_small",    "Qwen3-TTS-12Hz-0.6B-Base",        "Qwen3-TTS-12Hz-0.6B-Base",        "0.6B", "base"),
    ModelSpec("custom_small",  "Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS-12Hz-0.6B-CustomVoice", "0.6B", "custom"),
    ModelSpec("base",          "Qwen3-TTS-12Hz-1.7B-Base",        "Qwen3-TTS-12Hz-1.7B-Base",        "1.7B", "base"),
    ModelSpec("custom",        "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-CustomVoice", "1.7B", "custom"),
    ModelSpec("design",        "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "1.7B", "design"),
)

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in _REGISTRY}


def all_models() -> list[ModelSpec]:
    """Iteration order matches ``_REGISTRY``."""
    return list(_REGISTRY)


def model_keys() -> list[str]:
    return [spec.name for spec in _REGISTRY]


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        valid = ", ".join(model_keys())
        raise ValueError(f"Unknown model '{name}'. Available: {valid}") from None


def get_snapshot(spec_or_name: ModelSpec | str) -> Path:
    """Resolve a cached HF snapshot directory for a model.

    Previously this returned a ghost path on miss, which masked failures
    downstream. It now raises ``RuntimeError`` listing every path tried.
    """
    spec = spec_or_name if isinstance(spec_or_name, ModelSpec) else get_model(spec_or_name)
    tried: list[str] = []

    # 1. Preferred: huggingface_hub resolves the canonical snapshot dir.
    try:
        path = Path(snapshot_download(repo_id=spec.repo_id, local_files_only=True))
        log.debug("snapshot_download hit for %s: %s", spec.repo_id, path)
        return path
    except Exception as err:
        tried.append(f"snapshot_download({spec.repo_id}): {err}")

    # 2. Fallback: walk the HF cache layout directly. If the most recent
    #    snapshot subdir exists, use it.
    snap_dir = HF_HUB_ROOT / f"models--{HF_ORG}--{spec.repo}" / "snapshots"
    tried.append(str(snap_dir))
    if snap_dir.exists():
        snaps = [s for s in snap_dir.iterdir() if s.is_dir()]
        if snaps:
            # Newest-first so resuming an interrupted pull picks up the
            # latest commit instead of an abandoned partial.
            snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            log.debug("cache fallback hit for %s: %s", spec.repo_id, snaps[0])
            return snaps[0]

    raise RuntimeError(
        f"Model snapshot for '{spec.name}' ({spec.repo_id}) not found. Tried:\n  - "
        + "\n  - ".join(tried)
    )


@dataclass
class ModelConfig:
    """Per-project binding of a ModelSpec to a local destination."""

    name: str
    spec: ModelSpec
    source: Path
    dest: Path


def _probe_source(spec: ModelSpec) -> Path:
    """Return a best-guess source path for bootstrapping a ``ModelConfig``.

    Unlike :func:`get_snapshot` this never raises: if the model has not
    been fetched yet it returns the canonical HF cache directory that the
    downloader would populate. Callers rely on ``.exists()`` to drive the
    "download on demand" flow in ``cli.main._convert_internal``; surfacing
    a hard error at catalog-build time would be too early.
    """
    try:
        return get_snapshot(spec)
    except RuntimeError:
        return HF_HUB_ROOT / f"models--{HF_ORG}--{spec.repo}"


class Models:
    """Project-rooted view of the model catalog."""

    def __init__(self, project_root: Path, names: Iterable[str] | None = None):
        self.project_root = project_root
        keys = list(names) if names is not None else model_keys()
        self._models = [
            ModelConfig(
                name=k,
                spec=get_model(k),
                source=_probe_source(get_model(k)),
                dest=project_root / "models" / k,
            )
            for k in keys
        ]

    @property
    def all(self) -> list[ModelConfig]:
        return list(self._models)

    def by_name(self, name: str) -> ModelConfig:
        for model in self._models:
            if model.name == name:
                return model
        valid = ", ".join(m.name for m in self._models)
        raise ValueError(f"Unknown model '{name}'. Available: {valid}")
