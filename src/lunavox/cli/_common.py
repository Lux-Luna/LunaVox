"""Shared helpers for every ``lunavox`` subcommand.

Before the 4.2 refactor this lived as private underscore functions at
the top of ``main.py`` and was copy-pasted into convert / pull-model /
bootstrap. Factoring it out makes each command a thin wrapper around
``Engine`` + ``ResolvedConfig`` + a picker — no more state-threading
boilerplate per file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.prompt import Confirm

from lunavox.core.ui import console
from lunavox.model import all_models

from ._config import ResolvedConfig


@dataclass
class RuntimeState:
    """Session-wide context attached to every typer command.

    Kept tiny on purpose — anything tunable by the user lives in
    :class:`ResolvedConfig` (profile + env + flags), not here.
    """

    config: ResolvedConfig
    verbose: bool

    @property
    def project_root(self) -> Path:
        return self.config.project_root

    @property
    def latest_log(self) -> Path:
        return self.project_root / "logs" / "latest.log"


def state(ctx: typer.Context) -> RuntimeState:
    obj = ctx.obj
    if not isinstance(obj, RuntimeState):
        raise RuntimeError("CLI runtime state is missing")
    return obj


def confirm_or_fail(st: RuntimeState, prompt: str) -> bool:
    """Return True if the user confirms, fail loudly in non-interactive mode.

    ``--yes`` on the global callback short-circuits to True. Anywhere
    else we refuse to silently default rather than prompt someone's
    CI into a 5-GB download.
    """
    if st.config.yes:
        return True
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        raise RuntimeError(f"Non-interactive mode: {prompt} (re-run with --yes)")
    return Confirm.ask(prompt, default=True)


def pick_model(
    header: str,
    prompt: str,
    model: Optional[str],
    *,
    allow_all: bool = True,
) -> list[str]:
    """Resolve a ``--model`` flag or run the interactive picker.

    Returns an empty list if the user cancels. Raises ``RuntimeError``
    on invalid input. Used by every model-touching command so the
    selection UX is identical across the CLI.
    """
    if model:
        return [model]
    specs = all_models()
    console.print(f"\n[bold cyan]═══ {header} ═══[/]")
    for i, spec in enumerate(specs, 1):
        console.print(f"  [bold]{i}) {spec.display_name}[/]")
    if allow_all:
        console.print(f"  [bold]{len(specs) + 1}) ALL MODELS[/]")
    console.print("  [bold]0) Cancel[/]")

    choice = typer.prompt(f"\n{prompt}", default="1")
    if choice == "0":
        console.print("[warning]Cancelled.[/]")
        return []
    try:
        idx = int(choice) - 1
    except ValueError as err:
        raise RuntimeError("Invalid selection.") from err

    if allow_all and idx == len(specs):
        return [spec.name for spec in specs]
    if 0 <= idx < len(specs):
        return [specs[idx].name]
    raise RuntimeError("Invalid selection.")
