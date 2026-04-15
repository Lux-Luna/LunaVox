from __future__ import annotations

import os
from pathlib import Path

#: Marker file used to tag a directory as a valid LunaVox root in
#: deployment layouts that don't ship the source tree. Created by
#: ``lunavox build`` and by the Docker image entrypoint. The presence
#: of this file is an explicit opt-in signal — we deliberately don't
#: auto-detect by looking for ``build/lunavox.dll`` because a stale
#: build directory from a different project could otherwise hijack the
#: resolver.
#:
#: Contents are free-form text; we only care that the file exists.
DEPLOYMENT_MARKER = ".lunavox-root"


def _looks_like_project_root(path: Path) -> bool:
    """Return True if ``path`` is a valid LunaVox root.

    Two layouts are accepted:

    * **Dev checkout** — has ``CMakeLists.txt`` + ``src/``.
      This is the historical layout the resolver has always
      recognised; every `git clone` of LunaVox matches it.
    * **Deployment** — has a ``.lunavox-root`` marker file.
      Used by the Docker image (and by anyone shipping LunaVox as
      a standalone bundle) where the source tree isn't present but
      a built shared library + models/ + lib/ are. The marker is
      an explicit opt-in because "looks like a LunaVox build" is
      too ambiguous to infer automatically.
    """
    if (path / DEPLOYMENT_MARKER).exists():
        return True
    cmake = path / "CMakeLists.txt"
    src = path / "src"
    return cmake.exists() and src.exists()


def resolve_project_root(explicit_root: Path | None = None) -> Path:
    """Find and return the project root directory, creating 'models' and 'lib' if missing."""
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
    for sub in ["models", "lib"]:
        target = candidate / sub
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

    return candidate
