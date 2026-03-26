from __future__ import annotations

import os
from pathlib import Path


def _looks_like_project_root(path: Path) -> bool:
    """Check if the path contains core LunaVox project markers."""
    required = [
        path / "CMakeLists.txt",
        path / "src",
        path / "lib",
    ]
    return all(p.exists() for p in required)


def resolve_project_root(explicit_root: Path | None = None) -> Path:
    """Find and return the project root directory, creating 'models' if missing."""
    candidate = None

    if explicit_root is not None:
        candidate = explicit_root.resolve()
        if not _looks_like_project_root(candidate):
            raise RuntimeError(f"Invalid --project-root: {candidate}")
    else:
        env_root = os.environ.get("LUNAVOX_PROJECT_ROOT", "").strip()
        if env_root:
            env_candidate = Path(env_root).resolve()
            if _looks_like_project_root(env_candidate):
                candidate = env_candidate

        if candidate is None:
            cwd = Path.cwd().resolve()
            probe_paths = [cwd, *cwd.parents]
            for p in probe_paths:
                if _looks_like_project_root(p):
                    candidate = p
                    break

    if candidate is None:
        raise RuntimeError(
            "Could not locate LunaVox project root. Run inside the repository or pass --project-root."
        )

    # Ensure required runtime directories exist
    models_dir = candidate / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    return candidate

