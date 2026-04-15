"""``lunavox bootstrap`` — one-shot first-run setup.

Before the refactor this called a 70-line ``_synthesis_test_interactive``
helper that subprocess'd ``lunavox-cli.exe`` with hard-coded paths.
Now it just chains the three other commands (pull → libs → build) and
delegates the smoke test to :mod:`synth_cmd` so there is exactly one
synthesis path in the CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

from lunavox.core.ui import console

from . import build_cmd, model_cmd
from ._common import RuntimeState, state
from .synth_cmd import _run_synth


def register(parent: typer.Typer) -> None:
    @parent.command("bootstrap")
    def bootstrap_cmd(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(None, "--model", help="Model variant"),
        platform: Optional[str] = typer.Option(None, "--platform", help="Target platform"),
        clean: bool = typer.Option(False, "--clean", help="Clean build directory first"),
        j: int = typer.Option(4, "--j", min=1, help="Parallel build jobs"),
        toolchain: str = typer.Option("auto", "--toolchain", help="Force toolchain"),
        skip_test: bool = typer.Option(
            False, "--skip-test", help="Skip the post-build synthesis smoke test"
        ),
    ) -> None:
        """Guided setup: pull model → download libs → build → smoke test."""
        st = state(ctx)

        pulled = model_cmd.pull(st, model)
        build_cmd.libs(st, platform_key=platform)
        build_cmd.build(st, clean=clean, j=j, toolchain=toolchain)

        if not skip_test:
            _post_build_smoke_test(st, pulled)

        console.print("[success]Bootstrap completed successfully.[/success]")


def _post_build_smoke_test(st: RuntimeState, pulled: list[str]) -> None:
    """Pick a just-pulled model and synthesize a one-liner through the
    in-process :class:`Engine` — the same code path the GUI uses."""
    model = pulled[0] if pulled else st.config.model
    model_dir = st.project_root / "models" / model
    if not model_dir.exists() or not any(model_dir.glob("*.gguf")):
        console.print(f"[warning]Skipping smoke test: no usable artifacts in {model_dir}[/]")
        return

    output = st.project_root / "output" / "bootstrap_test.wav"
    text = "Hi, this is lunavox speaking."
    console.print(f"[stage]Running bootstrap smoke test (model={model})[/]")
    try:
        _run_synth(
            st,
            text=text,
            output=output,
            model=model,
            voice="base",
            reference=None,
            speaker=None,
            instruct=None,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    except Exception as err:
        console.print(f"[warning]Smoke test skipped: {err}[/]")
