from __future__ import annotations

import os
from pathlib import Path


def _looks_like_project_root(path: Path) -> bool:
    required = [
        path / "CMakeLists.txt",
        path / "src",
        path / "lib",
        path / "models",
    ]
    return all(p.exists() for p in required)


def resolve_project_root(explicit_root: Path | None = None) -> Path:
    if explicit_root is not None:
        candidate = explicit_root.resolve()
        if not _looks_like_project_root(candidate):
            raise RuntimeError(f"Invalid --project-root: {candidate}")
        return candidate

    env_root = os.environ.get("LUNAVOX_PROJECT_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).resolve()
        if _looks_like_project_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    probe_paths = [cwd, *cwd.parents]
    for candidate in probe_paths:
        if _looks_like_project_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate LunaVox project root. Run inside the repository or pass --project-root."
    )

