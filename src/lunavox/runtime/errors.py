"""Exception types raised by :mod:`lunavox.runtime`.

Kept in a tiny module so tests and callers can import the types
without pulling in ctypes, numpy, or the library loader.
"""

from __future__ import annotations


class LunavoxLibraryError(RuntimeError):
    """Raised when ``liblunavox`` cannot be located or loaded."""


class LunavoxSynthesisError(RuntimeError):
    """Raised when a C-side synthesize call returns NULL or the engine
    reports an error via ``lunavox_get_error``."""
