"""HTTP serving layer for LunaVox.

FastAPI, uvicorn, pydantic and prometheus-client ship with the base
install. The CLI ``lunavox serve`` command is the main entry;
:class:`EngineHolder` and :func:`create_app` are also importable for
embedding the HTTP surface inside a larger application.

Phase 5A scope: single in-process :class:`lunavox.runtime.Engine`
serialised via an ``asyncio.Lock`` so concurrent HTTP / WebSocket
clients queue politely on one GPU. Phase 5B will swap in a C++
BatchEngine for real continuous batching; the Python surface here
is designed to take that upgrade without a public API break.
"""

from __future__ import annotations
