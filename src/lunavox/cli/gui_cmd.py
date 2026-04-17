"""``lunavox gui`` — launch the desktop GUI.

The GUI lives in :mod:`lunavox.gui` and ships with the base install.
"""

from __future__ import annotations

import typer

from ._common import state


def register(parent: typer.Typer) -> None:
    @parent.command("gui")
    def gui_cmd(ctx: typer.Context) -> None:
        """Launch the LunaVox desktop GUI."""
        st = state(ctx)
        from lunavox.gui.app import LunaVoxApp

        LunaVoxApp(config=st.config).run()
