from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.prompt import Confirm

from .ui import console

DEPENDENCY_GROUPS: dict[str, list[tuple[str, str]]] = {
    "convert": [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("safetensors", "safetensors"),
        ("gguf", "gguf"),
        ("transformers", "transformers"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxscript", "onnxscript"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_modules(modules: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(name, pip_name) for name, pip_name in modules if not has_module(name)]


@dataclass
class DependencyPolicy:
    yes: bool = False
    no_install: bool = False


def ensure_dependency_group(group: str, policy: DependencyPolicy, project_root: Path | None = None) -> None:
    if group not in DEPENDENCY_GROUPS:
        raise RuntimeError(f"Unknown dependency group: {group}")

    missing = missing_modules(DEPENDENCY_GROUPS[group])
    if not missing:
        return

    missing_text = ", ".join(f"{name} ({pip_name})" for name, pip_name in missing)
    install_cmd = f'{sys.executable} -m pip install "lunavox[{group}]"'
    fallback_cmd = ""
    if project_root is not None and (project_root / "pyproject.toml").exists():
        fallback_cmd = f'{sys.executable} -m pip install ".[{group}]"'
    install_cmd_display = install_cmd.replace("[", "\\[").replace("]", "\\]")

    if policy.no_install:
        raise RuntimeError(
            "Missing required dependencies: "
            + missing_text
            + "\nAuto install disabled by --no-install.\nInstall manually:\n  "
            + install_cmd
            + (f"\nOr from local project:\n  {fallback_cmd}" if fallback_cmd else "")
        )

    non_interactive = not (sys.stdin.isatty() and sys.stdout.isatty())
    if non_interactive and not policy.yes:
        raise RuntimeError(
            "Missing required dependencies: "
            + missing_text
            + "\nNon-interactive mode detected. Re-run with --yes or install manually:\n  "
            + install_cmd
            + (f"\nOr from local project:\n  {fallback_cmd}" if fallback_cmd else "")
        )

    should_install = policy.yes
    if not should_install:
        should_install = Confirm.ask(
            f"Missing dependencies for '{group}': {missing_text}\nInstall now using `{install_cmd_display}`?",
            default=True,
        )

    if not should_install:
        raise RuntimeError(
            "Dependency installation declined.\nInstall manually:\n  "
            + install_cmd
            + (f"\nOr from local project:\n  {fallback_cmd}" if fallback_cmd else "")
        )

    env = os.environ.copy()
    try:
        # Prioritize local project installation if we are in the project root
        if fallback_cmd:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f".[{group}]"],
                check=True,
                env=env,
                cwd=str(project_root),
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"lunavox[{group}]"],
                check=True,
                env=env,
            )
    except subprocess.CalledProcessError:
        # Fallback to name-based install if local fail (or vice versa if we swapped them)
        if not fallback_cmd:
            raise
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"lunavox[{group}]"],
            check=True,
            env=env,
        )

    still_missing = missing_modules(DEPENDENCY_GROUPS[group])
    if still_missing:
        still_text = ", ".join(f"{name} ({pip_name})" for name, pip_name in still_missing)
        raise RuntimeError(
            "Dependency installation finished but modules are still missing: " + still_text
        )
    console.print(f"[success]Dependencies for '{group}' are ready.[/success]")
