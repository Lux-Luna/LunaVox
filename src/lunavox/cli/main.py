"""LunaVox CLI entry point — slim typer assembly.

The heavy lifting lives in the per-command modules (``model_cmd``,
``build_cmd``, ``synth_cmd``, …). This file's only jobs are:

1. Create the root ``app`` and register every subcommand.
2. On every invocation, build a :class:`ResolvedConfig` from the
   ``--profile`` flag + env vars + CLI flags, and attach a
   :class:`RuntimeState` to the typer context.
3. Provide ``run()`` so ``python -m lunavox`` and the
   ``[project.scripts] lunavox`` entry point share one function.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from lunavox.core import logging as lvlog
from lunavox.core.ui import console

from . import bootstrap_cmd, build_cmd, doctor_cmd, gui_cmd, model_cmd, synth_cmd
from ._common import RuntimeState
from ._config import load_config

app = typer.Typer(no_args_is_help=True, help="LunaVox unified CLI")
app.add_typer(model_cmd.app, name="model")
app.add_typer(build_cmd.app, name="build")
synth_cmd.register(app)
gui_cmd.register(app)
bootstrap_cmd.register(app)
doctor_cmd.register(app)


@app.callback()
def main(
    ctx: typer.Context,
    profile: Optional[str] = typer.Option(
        None, "--profile", help="Named profile from ~/.lunavox/config.toml"
    ),
    project_root: Optional[Path] = typer.Option(
        None, "--project-root", help="LunaVox project root"
    ),
    yes: bool = typer.Option(False, "--yes", help="Auto confirm install/download prompts"),
    no_install: bool = typer.Option(
        False, "--no-install", help="Never auto-install missing Python dependencies"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose command output"),
) -> None:
    if ctx.resilient_parsing:
        return
    config = load_config(
        profile=profile,
        project_root=project_root,
        yes=yes,
        no_install=no_install,
        verbose=verbose,
    )
    log_file = config.project_root / "logs" / "latest.log"
    lvlog.session_start(
        log_file,
        header=f"LunaVox CLI session start: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    ctx.obj = RuntimeState(config=config, verbose=verbose)


def run() -> int:
    try:
        app()
        return 0
    except Exception as err:
        console.print(str(err), style="error", markup=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
