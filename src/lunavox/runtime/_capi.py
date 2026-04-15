"""Raw ctypes binding for ``liblunavox``.

This module is deliberately narrow — it only knows how to:

* locate and dlopen the shared library (``load_library``)
* declare the ctypes mirrors of ``LunavoxSynthesisParams`` and
  ``LunavoxAudio`` so pyright/callers see a typed surface
* install ``argtypes`` / ``restype`` for every C function LunaVox uses

Anything richer (marshalling, numpy conversion, error formatting,
lifecycle) lives in :mod:`lunavox.runtime.engine`. Keeping this file
thin makes it trivial to audit against ``src/lunavox_c_api.h`` when
the C ABI changes.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from lunavox.core.platform import shared_lib_name
from lunavox.core.project import resolve_project_root

from .errors import LunavoxLibraryError

# ---------------------------------------------------------------------------
# ctypes struct layout — must stay in lock-step with src/lunavox_c_api.h
# ---------------------------------------------------------------------------


class CParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
        ("ref_text", ctypes.c_char_p),
    ]


class CAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("t_tokenize_ms", ctypes.c_int64),
        ("t_encode_ms", ctypes.c_int64),
        ("t_generate_ms", ctypes.c_int64),
        ("t_decode_ms", ctypes.c_int64),
        ("t_total_ms", ctypes.c_int64),
        ("audio_duration_ms", ctypes.c_int64),
        ("rtf", ctypes.c_float),
        ("rss_peak_bytes", ctypes.c_uint64),
        ("rss_end_bytes", ctypes.c_uint64),
    ]


# Matches enum LunavoxLogLevel in lunavox_c_api.h: (level, utf8 message, user ptr)
LOG_CALLBACK_T = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

# Streaming audio chunk callback — fired from the C++ decoder worker
# thread as each PCM slice becomes available. Signature matches
# LunavoxAudioChunkCallback in lunavox_c_api.h:
#   (const float * samples, int32_t n_samples, int32_t is_last, void * user_data)
AUDIO_CHUNK_CALLBACK_T = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_void_p,
)


# ---------------------------------------------------------------------------
# Library loader — process-wide, idempotent
# ---------------------------------------------------------------------------


_lib_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None


def _candidate_paths() -> list[Path]:
    """Return ordered search paths for ``liblunavox``.

    1. ``LUNAVOX_LIB_PATH`` env var (explicit override)
    2. ``<project_root>/build/{lunavox.dll,liblunavox.so,liblunavox.dylib}`` —
       the normal dev-loop location written by ``lunavox build``
    3. Installed alongside the python package (``site-packages/lunavox/runtime``)
       for future wheel packaging
    """
    name = shared_lib_name("lunavox")
    out: list[Path] = []

    env = os.environ.get("LUNAVOX_LIB_PATH", "").strip()
    if env:
        out.append(Path(env))

    try:
        root = resolve_project_root()
        out.append(root / "build" / name)
    except Exception:
        pass

    out.append(Path(__file__).parent / name)
    return out


def load_library() -> ctypes.CDLL:
    """Locate ``liblunavox``, dlopen it, and bind every symbol LunaVox
    uses. Subsequent calls return the same ``CDLL`` handle."""
    global _lib, _lib_path
    with _lib_lock:
        if _lib is not None:
            return _lib

        tried: list[str] = []
        for path in _candidate_paths():
            tried.append(str(path))
            if not path.exists():
                continue
            try:
                # Point the engine at the directory that holds llama.dll /
                # ggml.dll / onnxruntime*.dll. Without this env var the
                # C++ LlamaLibrary::ensure_loaded falls back to the Python
                # executable directory (miniconda), which doesn't have the
                # backend libraries and load_models silently fails partway.
                os.environ.setdefault("LUNAVOX_LIB_DIR", str(path.parent))

                # Windows needs the dependency DLL directory on the search
                # path before CDLL fires, otherwise the loader can't find
                # ggml/llama/onnxruntime siblings of liblunavox.dll.
                if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                    with contextlib.suppress(OSError):
                        os.add_dll_directory(str(path.parent))
                _lib = ctypes.CDLL(str(path))
                _lib_path = path
                _bind_symbols(_lib)
                return _lib
            except OSError as err:
                tried.append(f"  {path}: {err}")

        raise LunavoxLibraryError("Failed to load liblunavox. Tried:\n  - " + "\n  - ".join(tried))


def library_path() -> Optional[Path]:
    """Return the dlopen'd library path, loading on demand."""
    load_library()
    return _lib_path


def _bind_symbols(lib: ctypes.CDLL) -> None:
    """Declare argtypes / restype for every symbol called from Python.

    Explicit signatures avoid silent int truncation on 64-bit fields
    and let ctypes translate return pointers into real Python objects.
    """
    lib.lunavox_default_params.argtypes = [ctypes.POINTER(CParams)]
    lib.lunavox_default_params.restype = None

    lib.lunavox_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lunavox_create.restype = ctypes.c_void_p

    lib.lunavox_destroy.argtypes = [ctypes.c_void_p]
    lib.lunavox_destroy.restype = None

    lib.lunavox_is_loaded.argtypes = [ctypes.c_void_p]
    lib.lunavox_is_loaded.restype = ctypes.c_int

    lib.lunavox_sample_rate.argtypes = [ctypes.c_void_p]
    lib.lunavox_sample_rate.restype = ctypes.c_int32

    lib.lunavox_get_error.argtypes = [ctypes.c_void_p]
    lib.lunavox_get_error.restype = ctypes.c_char_p

    lib.lunavox_last_load_ms.argtypes = [ctypes.c_void_p]
    lib.lunavox_last_load_ms.restype = ctypes.c_int64

    lib.lunavox_last_warmup_ms.argtypes = [ctypes.c_void_p]
    lib.lunavox_last_warmup_ms.restype = ctypes.c_int64

    lib.lunavox_free_audio.argtypes = [ctypes.POINTER(CAudio)]
    lib.lunavox_free_audio.restype = None

    # Synthesize variants — all return LunavoxAudio*. The Python Engine
    # exposes only one method, but it dispatches to three of these based
    # on the Voice mode. (CLONE_SAMPLES / CLONE_EMBEDDING paths exist in
    # the C API for future use; they are not bound here because nothing
    # in Python calls them yet.)
    audio_ptr = ctypes.POINTER(CAudio)

    lib.lunavox_synthesize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(CParams),
    ]
    lib.lunavox_synthesize.restype = audio_ptr

    lib.lunavox_synthesize_with_voice_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(CParams),
    ]
    lib.lunavox_synthesize_with_voice_file.restype = audio_ptr

    lib.lunavox_synthesize_custom.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(CParams),
    ]
    lib.lunavox_synthesize_custom.restype = audio_ptr

    lib.lunavox_synthesize_design.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(CParams),
    ]
    lib.lunavox_synthesize_design.restype = audio_ptr

    # Streaming base-mode synthesis. Same return type as the non-streaming
    # variant; the chunk callback fires during the call for progressive
    # PCM delivery, and the cumulative audio is still returned at the end.
    lib.lunavox_synthesize_streaming.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(CParams),
        AUDIO_CHUNK_CALLBACK_T,
        ctypes.c_void_p,
    ]
    lib.lunavox_synthesize_streaming.restype = audio_ptr

    lib.lunavox_set_log_callback.argtypes = [LOG_CALLBACK_T, ctypes.c_void_p]
    lib.lunavox_set_log_callback.restype = None
