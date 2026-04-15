"""LunaVox desktop GUI.

Gated behind the ``[gui]`` optional extra — importing
:mod:`lunavox.gui.app` directly when customtkinter is missing will
raise ``ModuleNotFoundError``, and the CLI ``lunavox gui`` command
turns that into a clear "pip install lunavox[gui]" message.
"""

from __future__ import annotations
