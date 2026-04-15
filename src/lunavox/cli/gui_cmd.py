"""``lunavox gui`` — launch the desktop GUI.

The GUI lives in :mod:`lunavox.gui` and is gated behind the
``[gui]`` optional extra. This command only prints a clear "install
lunavox[gui]" hint when the extra is missing; it never silently
crashes on ``ModuleNotFoundError``.
"""

from __future__ import annotations

import typer

from lunavox.core.ui import console

from ._common import state


def register(parent: typer.Typer) -> None:
    @parent.command("gui")
    def gui_cmd(ctx: typer.Context) -> None:
        """Launch the LunaVox desktop GUI."""
        st = state(ctx)
        try:
            from lunavox.gui.app import LunaVoxApp  # pyright: ignore[reportMissingImports]
        except ImportError as err:
            console.print(
                "[error]The GUI extra is not installed. Run:[/]\n"
                '  [bold]pip install "lunavox[gui]"[/]\n'
                f"([dim]import failed: {err}[/])",
                markup=True,
            )
            raise typer.Exit(code=1) from err

        LunaVoxApp(config=st.config).run()
