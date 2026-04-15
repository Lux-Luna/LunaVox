"""``lunavox serve`` — HTTP / WebSocket serving layer.

Thin CLI wrapper around :func:`lunavox.serve.server.create_app`.
Gated behind the ``[serve]`` optional extra; missing extras print
a clear install hint instead of a ``ModuleNotFoundError`` stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from lunavox.core.ui import console

from ._common import state


def register(parent: typer.Typer) -> None:
    @parent.command("serve")
    def serve_cmd(
        ctx: typer.Context,
        host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
        port: int = typer.Option(8000, "--port", help="Bind port"),
        model: Optional[str] = typer.Option(
            None, "--model", help="Model directory name under models/ (override config)"
        ),
        log_level: str = typer.Option(
            "info", "--log-level", help="uvicorn log level: critical|error|warning|info|debug"
        ),
    ) -> None:
        """Start the HTTP / WebSocket serving layer."""
        st = state(ctx)

        try:
            import uvicorn  # pyright: ignore[reportMissingImports]

            from lunavox.serve.server import create_app  # pyright: ignore[reportMissingImports]
        except ImportError as err:
            console.print(
                "[error]The serve extra is not installed. Run:[/]\n"
                '  [bold]pip install "lunavox[serve]"[/]\n'
                f"([dim]import failed: {err}[/])",
                markup=True,
            )
            raise typer.Exit(code=1) from err

        resolved_model = model or st.config.model
        model_dir: Path = st.project_root / "models" / resolved_model
        if not model_dir.exists():
            console.print(
                f"[error]Model directory not found: {model_dir}. "
                f"Run `lunavox model pull --model {resolved_model}` first.[/]"
            )
            raise typer.Exit(code=1)

        console.print(
            f"[stage]Starting LunaVox server at [bold]http://{host}:{port}[/] "
            f"(model=[bold]{resolved_model}[/], threads={st.config.n_threads})[/]"
        )

        app = create_app(model_dir, n_threads=st.config.n_threads)
        uvicorn.run(app, host=host, port=port, log_level=log_level)
