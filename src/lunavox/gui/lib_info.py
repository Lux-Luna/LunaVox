"""Helpers to surface installed native-runtime backends in the GUI.

Reads ``<project>/lib/metadata.json``, which the lib downloader writes
each time a backend is installed. Both the Settings view (for a quick
"what's active" readout) and the Library view (for the per-row
download UI) consume this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BackendInfo:
    """One lib's currently-installed backend, as written by the downloader."""

    label: str  # "vulkan", "DmlExecutionProvider", etc.
    version: str
    platform: str
    installed_at: str


def read_lib_metadata(project_root: Path) -> dict[str, BackendInfo]:
    """Return ``{lib_name: BackendInfo}`` for every entry in ``lib/metadata.json``.

    Returns an empty dict if the file is missing or malformed — the
    GUI then renders "(not installed)" for each lib.
    """
    path = project_root / "lib" / "metadata.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, BackendInfo] = {}
    for lib_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        # The downloader writes "provider" for onnx and "backend" for llama;
        # treat them the same in the UI.
        label = str(entry.get("provider") or entry.get("backend") or "unknown")
        out[lib_name] = BackendInfo(
            label=label,
            version=str(entry.get("version", "")),
            platform=str(entry.get("platform", "")),
            installed_at=str(entry.get("installed_at", "")),
        )
    return out


def lib_backend(project_root: Path, lib_name: str) -> Optional[BackendInfo]:
    return read_lib_metadata(project_root).get(lib_name)
