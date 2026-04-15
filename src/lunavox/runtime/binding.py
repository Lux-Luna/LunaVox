"""ctypes binding for ``liblunavox`` — the C ABI facade around
``lunavox::Engine``.

Design:

- One process-wide library handle, lazily loaded the first time any
  consumer touches the module. Loading is idempotent.
- ``Engine`` is a thin RAII wrapper. ``__enter__``/``__exit__`` and
  ``close()`` free the C handle. Forgetting to close leaks the C++
  engine, so the GUI and scripts should use ``with`` where possible.
- Synthesis entry points return a :class:`SynthesisResult` dataclass
  whose ``audio`` field is a ``numpy.ndarray`` copy of the PCM buffer.
  The C-side ``LunavoxAudio*`` is freed before returning, so callers
  never juggle raw pointers.
- ``SynthesisStats`` mirrors the fields the C API embeds in
  ``LunavoxAudio`` (tokenize/encode/generate/decode timings, RTF, peak
  RSS). Consumers get structured data instead of regex-parsing stdout.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from lunavox.core.platform import shared_lib_name
from lunavox.core.project import resolve_project_root


class LunavoxLibraryError(RuntimeError):
    """Raised when ``liblunavox`` cannot be located or loaded."""


class LunavoxSynthesisError(RuntimeError):
    """Raised when the C engine returns NULL from a synthesize call."""


# ---------------------------------------------------------------------------
# ctypes struct layout — must stay in lock-step with src/lunavox_c_api.h
# ---------------------------------------------------------------------------


class _CParams(ctypes.Structure):
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


class _CAudio(ctypes.Structure):
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


# Matches enum LunavoxLogLevel in lunavox_c_api.h
_LOG_CALLBACK_T = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------


_lib_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None


def _candidate_lib_paths() -> list[Path]:
    """Return ordered search paths for ``liblunavox``.

    1. ``LUNAVOX_LIB_PATH`` env var (explicit override).
    2. ``<project_root>/build/{lunavox.dll,liblunavox.so,liblunavox.dylib}``
       — the normal dev-loop location written by ``lunavox build``.
    3. Installed alongside the python package (``site-packages/lunavox/runtime``)
       for future wheel packaging.
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


def _load_library() -> ctypes.CDLL:
    global _lib, _lib_path
    with _lib_lock:
        if _lib is not None:
            return _lib

        tried: list[str] = []
        for path in _candidate_lib_paths():
            tried.append(str(path))
            if not path.exists():
                continue
            try:
                # Point the engine at the directory that holds llama.dll /
                # ggml.dll / onnxruntime*.dll. LlamaLibrary::ensure_loaded
                # reads LUNAVOX_LIB_DIR first; without it the engine falls
                # back to the Python executable directory (miniconda), which
                # doesn't have the backend libraries and load_models silently
                # fails partway through.
                os.environ.setdefault("LUNAVOX_LIB_DIR", str(path.parent))

                # The C++ shared lib depends on the runtime DLLs in the
                # same build/ directory (ggml, llama, onnxruntime, DirectML,
                # etc.). On Windows the loader needs that directory on the
                # DLL search path before CDLL is called.
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


def _bind_symbols(lib: ctypes.CDLL) -> None:
    """Install the argtypes/restype for every function we call.

    Keeping this explicit avoids silent int truncation on 64-bit fields
    and lets ctypes translate return pointers into real Python objects.
    """
    lib.lunavox_default_params.argtypes = [ctypes.POINTER(_CParams)]
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

    lib.lunavox_free_audio.argtypes = [ctypes.POINTER(_CAudio)]
    lib.lunavox_free_audio.restype = None

    # Synthesize variants — all return LunavoxAudio*.
    audio_ptr = ctypes.POINTER(_CAudio)

    lib.lunavox_synthesize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_CParams)]
    lib.lunavox_synthesize.restype = audio_ptr

    lib.lunavox_synthesize_with_voice_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CParams),
    ]
    lib.lunavox_synthesize_with_voice_file.restype = audio_ptr

    lib.lunavox_synthesize_with_voice_samples.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(_CParams),
    ]
    lib.lunavox_synthesize_with_voice_samples.restype = audio_ptr

    lib.lunavox_synthesize_with_embedding.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(_CParams),
    ]
    lib.lunavox_synthesize_with_embedding.restype = audio_ptr

    lib.lunavox_synthesize_custom.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CParams),
    ]
    lib.lunavox_synthesize_custom.restype = audio_ptr

    lib.lunavox_synthesize_design.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CParams),
    ]
    lib.lunavox_synthesize_design.restype = audio_ptr

    lib.lunavox_extract_embedding_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.lunavox_extract_embedding_file.restype = ctypes.c_int32

    lib.lunavox_set_log_callback.argtypes = [_LOG_CALLBACK_T, ctypes.c_void_p]
    lib.lunavox_set_log_callback.restype = None


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class SynthesisMode(Enum):
    BASE = "base"
    CLONE_FILE = "clone_file"
    CLONE_SAMPLES = "clone_samples"
    CLONE_EMBEDDING = "clone_embedding"
    CUSTOM = "custom"
    DESIGN = "design"


@dataclass
class SynthesisStats:
    """Per-run timing and memory stats echoed from the C engine."""

    t_tokenize_ms: int = 0
    t_encode_ms: int = 0
    t_generate_ms: int = 0
    t_decode_ms: int = 0
    t_total_ms: int = 0
    audio_duration_ms: int = 0
    rtf: float = 0.0
    rss_peak_bytes: int = 0
    rss_end_bytes: int = 0


@dataclass
class SynthesisResult:
    """Structured output of a single synthesize call.

    ``audio`` is a ``numpy.float32`` array in [-1, 1] (mono). ``stats``
    holds the timing breakdown. ``mode`` records which synthesize
    variant produced the audio so downstream telemetry does not need to
    guess from kwargs.
    """

    audio: (
        object  # numpy.ndarray; type kept loose so numpy is a runtime dep, not an import-time one
    )
    sample_rate: int
    stats: SynthesisStats
    mode: SynthesisMode


@dataclass
class SynthesisParams:
    """Python-side mirror of ``LunavoxSynthesisParams``.

    Kept as a dataclass (not a direct ctypes struct) so callers use
    native Python types and ``Engine.synthesize`` handles the
    ctypes marshalling and lifetime of the ``ref_text`` buffer.
    """

    max_audio_tokens: int = 0
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 50
    n_threads: int = 4
    repetition_penalty: float = 1.05
    language_id: int = -1
    ref_text: Optional[str] = None


def default_params() -> SynthesisParams:
    """Return defaults that match ``lunavox_default_params`` in the C API."""
    return SynthesisParams()


# ---------------------------------------------------------------------------
# Internal marshalling
# ---------------------------------------------------------------------------


def _to_c_params(p: Optional[SynthesisParams]) -> tuple[_CParams, list[bytes]]:
    """Copy a ``SynthesisParams`` into a ``_CParams``.

    Returns both the struct *and* the list of backing byte buffers that
    ``ref_text`` points into. Callers must keep the buffer list alive for
    the duration of the C call so the ``c_char_p`` field is not freed
    underneath the engine.
    """
    cp = _CParams()
    _load_library().lunavox_default_params(ctypes.byref(cp))
    if p is None:
        return cp, []

    cp.max_audio_tokens = int(p.max_audio_tokens)
    cp.temperature = float(p.temperature)
    cp.top_p = float(p.top_p)
    cp.top_k = int(p.top_k)
    cp.n_threads = int(p.n_threads)
    cp.repetition_penalty = float(p.repetition_penalty)
    cp.language_id = int(p.language_id)

    held: list[bytes] = []
    if p.ref_text:
        buf = p.ref_text.encode("utf-8")
        held.append(buf)
        cp.ref_text = buf
    else:
        cp.ref_text = None
    return cp, held


def _consume_audio(ptr, mode: SynthesisMode) -> SynthesisResult:
    """Copy the PCM + stats out of a C ``LunavoxAudio*`` and free it."""
    import numpy as np  # deferred so importing the module doesn't cost numpy

    if not ptr:
        raise LunavoxSynthesisError("synthesize returned NULL")

    audio = ptr.contents
    n = int(audio.n_samples)
    # ctypes' .contents[:n] slicing copies into a Python list; wrapping
    # via numpy.ctypeslib + copy() avoids the intermediate list and is
    # noticeably faster for long-form output.
    pcm_ptr = ctypes.cast(audio.samples, ctypes.POINTER(ctypes.c_float * n))
    pcm = np.frombuffer(pcm_ptr.contents, dtype=np.float32).copy()

    stats = SynthesisStats(
        t_tokenize_ms=int(audio.t_tokenize_ms),
        t_encode_ms=int(audio.t_encode_ms),
        t_generate_ms=int(audio.t_generate_ms),
        t_decode_ms=int(audio.t_decode_ms),
        t_total_ms=int(audio.t_total_ms),
        audio_duration_ms=int(audio.audio_duration_ms),
        rtf=float(audio.rtf),
        rss_peak_bytes=int(audio.rss_peak_bytes),
        rss_end_bytes=int(audio.rss_end_bytes),
    )
    result = SynthesisResult(
        audio=pcm,
        sample_rate=int(audio.sample_rate),
        stats=stats,
        mode=mode,
    )
    _load_library().lunavox_free_audio(ptr)
    return result


# ---------------------------------------------------------------------------
# Log callback
# ---------------------------------------------------------------------------


# We hold a module-level reference to the ctypes callback so the trampoline
# object is not garbage collected while the C engine still has the pointer.
_log_cb_holder: Optional[Callable] = None
_log_cb_ctype: Optional[_LOG_CALLBACK_T] = None


LogCallback = Callable[[int, str], None]


def set_log_callback(cb: Optional[LogCallback]) -> None:
    """Install a Python-side log sink that receives every C++ log line.

    ``cb`` is called as ``cb(level: int, message: str)`` where ``level``
    is the raw ``LunavoxLogLevel`` (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR,
    4=USER). Passing ``None`` removes the current callback.
    """
    global _log_cb_holder, _log_cb_ctype
    lib = _load_library()

    if cb is None:
        lib.lunavox_set_log_callback(_LOG_CALLBACK_T(0), None)
        _log_cb_holder = None
        _log_cb_ctype = None
        return

    def _trampoline(level: int, message_ptr, _user: Optional[int]) -> None:
        if message_ptr is None:
            return
        try:
            text = ctypes.string_at(message_ptr).decode("utf-8", errors="replace")
        except Exception:
            return
        # Never let a Python exception escape across the C boundary.
        with contextlib.suppress(Exception):
            cb(int(level), text)

    _log_cb_ctype = _LOG_CALLBACK_T(_trampoline)
    _log_cb_holder = cb
    lib.lunavox_set_log_callback(_log_cb_ctype, None)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """RAII wrapper around ``LunavoxEngine*``.

    Construction loads models synchronously (blocking on GPU warmup).
    Call :meth:`close` or use the ``with`` statement to release the
    underlying C++ engine. Methods raise ``LunavoxSynthesisError`` on
    failure with the error message pulled from ``lunavox_get_error``.
    """

    def __init__(self, model_dir: Path | str, n_threads: int = 4):
        self._lib = _load_library()
        self._handle: Optional[int] = None
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        handle = self._lib.lunavox_create(
            str(self.model_dir).encode("utf-8"),
            int(n_threads),
        )
        if not handle:
            # Pull the error the C API stashed during the failed create —
            # passing handle=None retrieves the process-wide last-error.
            err_ptr = self._lib.lunavox_get_error(ctypes.c_void_p(0))
            detail = err_ptr.decode("utf-8", errors="replace") if err_ptr else "unknown error"
            raise LunavoxSynthesisError(f"lunavox_create failed for {self.model_dir}: {detail}")
        self._handle = handle

    # --- lifecycle -----------------------------------------------------

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._lib.lunavox_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    # --- introspection -------------------------------------------------

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise LunavoxSynthesisError("Engine has been closed")
        return self._handle

    def is_loaded(self) -> bool:
        return bool(self._lib.lunavox_is_loaded(self.handle))

    @property
    def sample_rate(self) -> int:
        return int(self._lib.lunavox_sample_rate(self.handle))

    @property
    def last_load_ms(self) -> int:
        return int(self._lib.lunavox_last_load_ms(self.handle))

    @property
    def last_warmup_ms(self) -> int:
        return int(self._lib.lunavox_last_warmup_ms(self.handle))

    def _raise_for_null(self, ptr) -> None:
        if ptr:
            return
        err_ptr = self._lib.lunavox_get_error(self.handle)
        msg = err_ptr.decode("utf-8", errors="replace") if err_ptr else "unknown error"
        raise LunavoxSynthesisError(msg)

    # --- synthesis -----------------------------------------------------

    def synthesize(
        self,
        text: str,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        cp, _held = _to_c_params(params)
        ptr = self._lib.lunavox_synthesize(
            self.handle,
            text.encode("utf-8"),
            ctypes.byref(cp),
        )
        self._raise_for_null(ptr)
        return _consume_audio(ptr, SynthesisMode.BASE)

    def synthesize_with_voice_file(
        self,
        text: str,
        reference_path: Path | str,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        cp, _held = _to_c_params(params)
        ptr = self._lib.lunavox_synthesize_with_voice_file(
            self.handle,
            text.encode("utf-8"),
            str(reference_path).encode("utf-8"),
            ctypes.byref(cp),
        )
        self._raise_for_null(ptr)
        return _consume_audio(ptr, SynthesisMode.CLONE_FILE)

    def synthesize_custom(
        self,
        text: str,
        speaker: str,
        instruct: str = "",
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        cp, _held = _to_c_params(params)
        ptr = self._lib.lunavox_synthesize_custom(
            self.handle,
            text.encode("utf-8"),
            speaker.encode("utf-8"),
            instruct.encode("utf-8") if instruct else b"",
            ctypes.byref(cp),
        )
        self._raise_for_null(ptr)
        return _consume_audio(ptr, SynthesisMode.CUSTOM)

    def synthesize_design(
        self,
        text: str,
        instruct: str,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        if not instruct.strip():
            raise LunavoxSynthesisError("Voice Design mode requires non-empty instruct text")
        cp, _held = _to_c_params(params)
        ptr = self._lib.lunavox_synthesize_design(
            self.handle,
            text.encode("utf-8"),
            instruct.encode("utf-8"),
            ctypes.byref(cp),
        )
        self._raise_for_null(ptr)
        return _consume_audio(ptr, SynthesisMode.DESIGN)


def library_path() -> Optional[Path]:
    """Return the currently loaded library path, loading on demand."""
    _load_library()
    return _lib_path
