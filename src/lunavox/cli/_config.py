"""User-facing configuration with profile support.

Precedence (highest wins):

1. CLI flags (``--temperature 0.9``, ``--backend cuda``)
2. Environment variables (``LUNAVOX_BACKEND``, ``LUNAVOX_MODEL``)
3. The selected ``[profile.<name>]`` table from ``config.toml``
4. The ``[default]`` table from ``config.toml``
5. Hardcoded defaults (``DEFAULT_CONFIG`` below)

``~/.lunavox/config.toml`` is the standard location; users can point
``LUNAVOX_CONFIG`` at a different file. The loader never crashes on
a missing config — it falls back to defaults, which is what the
first-run CLI experience should feel like.

We avoid pulling in pydantic here. The config surface is small enough
that a dataclass + a merge function is clearer than a framework, and
every new field stays one line instead of six.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — exercised on 3.10 CI jobs
    import tomli as tomllib

from lunavox.core.project import resolve_project_root

DEFAULT_CONFIG_PATH = Path.home() / ".lunavox" / "config.toml"


@dataclass
class ResolvedConfig:
    """The final, merged view every command actually reads.

    ``project_root`` and ``yes`` / ``no_install`` / ``verbose`` are
    shell-session state — they come only from CLI flags. Everything
    else (model, backend, n_threads, synthesis params) layers
    defaults → profile → env → CLI flags.
    """

    # Session-level flags — CLI-only, never in the config file.
    project_root: Path = field(default_factory=resolve_project_root)
    yes: bool = False
    no_install: bool = False
    verbose: bool = False

    # User-tunable defaults.
    model: str = "base_small"
    backend: str = "auto"
    n_threads: int = 4

    # Synthesis defaults — mirror lunavox.runtime.SynthesisParams so a
    # config file can pin a preferred sampler without code changes.
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    language_id: int = -1

    # The profile the user actually picked, for echo / debug.
    profile_name: Optional[str] = None


# Environment variables we honor. Mapped to ResolvedConfig field names.
_ENV_PREFIX = "LUNAVOX_"
_ENV_KEYS = {
    "MODEL": ("model", str),
    "BACKEND": ("backend", str),
    "N_THREADS": ("n_threads", int),
    "TEMPERATURE": ("temperature", float),
    "TOP_P": ("top_p", float),
    "TOP_K": ("top_k", int),
}


def _coerce(value: Any, to_type: type) -> Any:
    """Convert a raw toml/env value to the declared dataclass type."""
    if to_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    return to_type(value)


def _apply_table(cfg: ResolvedConfig, table: dict[str, Any]) -> None:
    """Merge keys from one TOML table into the dataclass in place."""
    by_name = {f.name: f.type for f in fields(cfg) if f.name != "project_root"}
    for key, value in table.items():
        if key not in by_name:
            continue
        t = by_name[key]
        # ``type`` here is the string annotation — resolve the handful
        # we actually support rather than importing typing machinery.
        if t in ("int", int):
            setattr(cfg, key, int(value))
        elif t in ("float", float):
            setattr(cfg, key, float(value))
        elif t in ("bool", bool):
            setattr(cfg, key, _coerce(value, bool))
        else:  # str and Optional[str]
            setattr(cfg, key, None if value is None else str(value))


def _config_file() -> Optional[Path]:
    """Return the config file to read, if any.

    ``LUNAVOX_CONFIG`` overrides; otherwise we fall back to the
    XDG-ish default under ``~/.lunavox/``. Missing file → ``None``.
    """
    override = os.environ.get("LUNAVOX_CONFIG", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None


def load_config(
    profile: Optional[str] = None,
    *,
    project_root: Optional[Path] = None,
    yes: bool = False,
    no_install: bool = False,
    verbose: bool = False,
    overrides: Optional[dict[str, Any]] = None,
) -> ResolvedConfig:
    """Resolve the effective config for one CLI invocation.

    ``overrides`` is the dict of CLI flags the caller explicitly set —
    keys that match :class:`ResolvedConfig` fields win over every
    other layer. Passing ``None`` for a value skips the override, so
    ``typer`` defaults don't clobber profile settings.
    """
    cfg = ResolvedConfig(
        project_root=resolve_project_root(project_root),
        yes=yes,
        no_install=no_install,
        verbose=verbose,
        profile_name=profile,
    )

    path = _config_file()
    if path is not None:
        with path.open("rb") as f:
            raw = tomllib.load(f)
        default_table = raw.get("default", {})
        if isinstance(default_table, dict):
            _apply_table(cfg, default_table)
        if profile:
            profiles_root = raw.get("profile", {})
            if isinstance(profiles_root, dict) and profile in profiles_root:
                _apply_table(cfg, profiles_root[profile])
            elif profile:
                raise RuntimeError(
                    f"Profile '{profile}' not found in {path}. "
                    f"Available: {', '.join(profiles_root) or '(none)'}"
                )

    for env_suffix, (attr, caster) in _ENV_KEYS.items():
        raw_val = os.environ.get(_ENV_PREFIX + env_suffix, "").strip()
        if raw_val:
            # A bad env var shouldn't crash the CLI — just ignore it.
            with contextlib.suppress(TypeError, ValueError):
                setattr(cfg, attr, caster(raw_val))

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return cfg
