"""The high-level :class:`Engine` wrapper.

``Engine`` owns a ``LunavoxEngine*`` handle and exposes exactly one
synthesis entry point:

.. code-block:: python

    with Engine("models/base_small") as engine:
        result = engine.synthesize("Hello, world!", voice=Voice.base())
        save_wav("out.wav", result.audio, result.sample_rate)

All the mode-specific plumbing (clone-from-file, custom speaker,
design-from-instruction) is carried by the :class:`Voice` argument.
Adding a new voice mode is a two-file change — one classmethod in
:mod:`voice` plus one branch in :meth:`Engine._dispatch`.
"""

from __future__ import annotations

import contextlib
import ctypes
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Optional, Union

from . import _capi
from .errors import LunavoxSynthesisError
from .params import (
    MemStats,
    SynthesisChunk,
    SynthesisMode,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
)
from .voice import Voice

LogCallback = Callable[[int, str], None]


def _to_c_params(p: Optional[SynthesisParams]) -> tuple[_capi.CParams, list[bytes]]:
    """Copy a :class:`SynthesisParams` into a ``CParams`` struct.

    Returns the struct *and* the list of backing byte buffers that
    ``ref_text`` points into. Callers must keep the buffer list alive
    for the duration of the C call so the ``c_char_p`` field stays
    valid.
    """
    cp = _capi.CParams()
    _capi.load_library().lunavox_default_params(ctypes.byref(cp))
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


def _consume_audio(ptr: Any, mode: SynthesisMode) -> SynthesisResult:
    """Copy PCM + stats out of a C ``LunavoxAudio*`` and free it."""
    import numpy as np  # deferred so importing the module doesn't cost numpy

    if not ptr:
        raise LunavoxSynthesisError("synthesize returned NULL")

    audio = ptr.contents
    n = int(audio.n_samples)
    pcm_ptr = ctypes.cast(audio.samples, ctypes.POINTER(ctypes.c_float * n))
    # ctypes.Array is a PEP 3118 buffer producer but pyright's stubs
    # don't expose it as one; memoryview() feeds numpy through the
    # buffer protocol without a Python-list detour.
    pcm = np.frombuffer(memoryview(pcm_ptr.contents), dtype=np.float32).copy()

    mem = MemStats(
        rss_start_bytes=int(audio.mem.rss_start_bytes),
        rss_end_bytes=int(audio.mem.rss_end_bytes),
        rss_peak_bytes=int(audio.mem.rss_peak_bytes),
        vram_start_bytes=int(audio.mem.vram_start_bytes),
        vram_end_bytes=int(audio.mem.vram_end_bytes),
        vram_peak_bytes=int(audio.mem.vram_peak_bytes),
        vram_measured=bool(audio.mem.vram_measured),
    )
    stats = SynthesisStats(
        t_tokenize_ms=int(audio.t_tokenize_ms),
        t_encode_ms=int(audio.t_encode_ms),
        t_generate_ms=int(audio.t_generate_ms),
        t_decode_ms=int(audio.t_decode_ms),
        t_total_ms=int(audio.t_total_ms),
        audio_duration_ms=int(audio.audio_duration_ms),
        rtf=float(audio.rtf),
        mem=mem,
    )
    result = SynthesisResult(
        audio=pcm,
        sample_rate=int(audio.sample_rate),
        stats=stats,
        mode=mode,
    )
    _capi.load_library().lunavox_free_audio(ptr)
    return result


class Engine:
    """RAII wrapper around ``LunavoxEngine*``.

    Construction loads models synchronously (blocking on GPU warmup).
    Call :meth:`close` or use a ``with`` block to release the
    underlying C++ engine. Methods raise :class:`LunavoxSynthesisError`
    on failure with the message pulled from ``lunavox_get_error``.
    """

    def __init__(self, model_dir: Union[str, Path], n_threads: int = 4):
        self._lib = _capi.load_library()
        self._handle: Optional[int] = None
        self._log_ctype: Any = None  # keeps the ctypes trampoline alive
        self._log_cb: Optional[LogCallback] = None
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        handle = self._lib.lunavox_create(
            str(self.model_dir).encode("utf-8"),
            int(n_threads),
        )
        if not handle:
            # Pull the last error from the global slot — passing NULL
            # retrieves the process-wide error stashed during create.
            err_ptr = self._lib.lunavox_get_error(ctypes.c_void_p(0))
            detail = err_ptr.decode("utf-8", errors="replace") if err_ptr else "unknown error"
            raise LunavoxSynthesisError(f"lunavox_create failed for {self.model_dir}: {detail}")
        self._handle = handle

    # --- lifecycle -----------------------------------------------------

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._lib.lunavox_destroy(self._handle)
            self._handle = None
            # Drop the log trampoline so the C engine cannot call back
            # into a freed Python wrapper.
            self._log_ctype = None
            self._log_cb = None

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

    # --- logging -------------------------------------------------------

    def on_log(self, cb: Optional[LogCallback]) -> None:
        """Install a Python-side log sink that receives every C++ log line.

        ``cb`` is invoked as ``cb(level: int, message: str)`` where
        ``level`` is the raw ``LunavoxLogLevel`` (0=DEBUG, 1=INFO,
        2=WARN, 3=ERROR, 4=USER). Passing ``None`` removes the current
        callback. The Engine holds the ctypes trampoline so the C
        engine never dereferences a freed Python object.
        """
        if cb is None:
            self._lib.lunavox_set_log_callback(_capi.LOG_CALLBACK_T(0), None)
            self._log_ctype = None
            self._log_cb = None
            return

        def _trampoline(level: int, message_ptr: Any, _user: Optional[int]) -> None:
            if message_ptr is None:
                return
            try:
                text = ctypes.string_at(message_ptr).decode("utf-8", errors="replace")
            except Exception:
                return
            # Never let a Python exception escape across the C boundary.
            with contextlib.suppress(Exception):
                cb(int(level), text)

        self._log_ctype = _capi.LOG_CALLBACK_T(_trampoline)
        self._log_cb = cb
        self._lib.lunavox_set_log_callback(self._log_ctype, None)

    # --- synthesis -----------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        """Synthesise ``text`` into audio using ``voice``.

        ``voice`` defaults to :func:`Voice.base` (the model's built-in
        speaker). ``params`` defaults to :func:`SynthesisParams`
        defaults, which match ``lunavox_default_params`` in the C API.
        The dispatch table below is the only place that knows which C
        symbol each mode maps to — voice-side changes never ripple out.
        """
        v = voice if voice is not None else Voice.base()
        cp, _held = _to_c_params(params)
        ptr = self._dispatch(text, v, cp)
        self._raise_for_null(ptr)
        return _consume_audio(ptr, v.mode)

    def _dispatch(self, text: str, voice: Voice, cp: _capi.CParams) -> Any:
        """Route one synthesis call to the right C symbol.

        This is the *only* place in LunaVox that knows the
        voice-mode → C-symbol mapping. Adding a mode means adding one
        branch here; every caller above goes through
        :meth:`synthesize` and stays untouched.
        """
        text_b = text.encode("utf-8")
        if voice.mode is SynthesisMode.BASE:
            return self._lib.lunavox_synthesize(self.handle, text_b, ctypes.byref(cp))
        if voice.mode is SynthesisMode.CLONE_FILE:
            if voice.reference_path is None:
                raise LunavoxSynthesisError("Voice.clone_file requires reference_path")
            return self._lib.lunavox_synthesize_with_voice_file(
                self.handle,
                text_b,
                str(voice.reference_path).encode("utf-8"),
                ctypes.byref(cp),
            )
        if voice.mode is SynthesisMode.CUSTOM:
            if not voice.speaker:
                raise LunavoxSynthesisError("Voice.custom requires speaker id")
            return self._lib.lunavox_synthesize_custom(
                self.handle,
                text_b,
                voice.speaker.encode("utf-8"),
                (voice.instruct or "").encode("utf-8"),
                ctypes.byref(cp),
            )
        if voice.mode is SynthesisMode.DESIGN:
            if not (voice.instruct and voice.instruct.strip()):
                raise LunavoxSynthesisError("Voice.design requires non-empty instruct")
            return self._lib.lunavox_synthesize_design(
                self.handle,
                text_b,
                voice.instruct.encode("utf-8"),
                ctypes.byref(cp),
            )
        raise LunavoxSynthesisError(f"Unsupported voice mode: {voice.mode}")

    def _raise_for_null(self, ptr: Any) -> None:
        if ptr:
            return
        err_ptr = self._lib.lunavox_get_error(self.handle)
        msg = err_ptr.decode("utf-8", errors="replace") if err_ptr else "unknown error"
        raise LunavoxSynthesisError(msg)

    # --- streaming synthesis -------------------------------------------

    def synthesize_stream(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> Iterator[SynthesisChunk]:
        """Stream PCM chunks as they become available during synthesis.

        Yields :class:`SynthesisChunk` objects; the terminal chunk has
        ``is_last=True`` and carries the full :class:`SynthesisStats`
        for the run. All four voice modes are supported — dispatch is
        the same idea as :meth:`_dispatch` for the one-shot path, but
        the call goes into the ``_streaming`` C API variant with a
        chunk callback attached.

        Implementation detail — the C engine fires its audio chunk
        callback from the decoder worker thread. That callback runs
        under the Python GIL (ctypes trampolines acquire it) but the
        Python code in the callback must stay short: we only copy the
        slice into a numpy array and push it onto a ``queue.Queue``.
        A background thread drives the C streaming call while the
        generator drains the queue on the caller's thread.
        """
        import numpy as np

        v = voice if voice is not None else Voice.base()
        cp, _held = _to_c_params(params)
        chunk_queue: queue.Queue[Any] = queue.Queue(maxsize=64)
        SENTINEL = object()  # noqa: N806

        def _on_chunk(
            samples_ptr: Any,
            n_samples: int,
            is_last: int,
            _user: Optional[int],
        ) -> None:
            # Copy the slice into a standalone ndarray before returning:
            # the C pointer is only valid for the duration of this call.
            if n_samples > 0 and samples_ptr:
                arr_type = ctypes.c_float * n_samples
                buf = ctypes.cast(samples_ptr, ctypes.POINTER(arr_type))
                audio = np.frombuffer(memoryview(buf.contents), dtype=np.float32).copy()
            else:
                audio = np.empty(0, dtype=np.float32)
            chunk_queue.put((audio, bool(is_last)))

        cb_trampoline = _capi.AUDIO_CHUNK_CALLBACK_T(_on_chunk)
        sr_holder: dict[str, int] = {"rate": 0}
        stats_holder: dict[str, Optional[SynthesisStats]] = {"stats": None}
        err_holder: dict[str, Optional[BaseException]] = {"err": None}

        def _worker() -> None:
            try:
                ptr = self._dispatch_stream(text, v, cp, cb_trampoline)
                self._raise_for_null(ptr)
                # The full cumulative audio is still returned here; we
                # only keep the stats since callers consumed the chunks
                # progressively through the queue.
                result = _consume_audio(ptr, v.mode)
                sr_holder["rate"] = result.sample_rate
                stats_holder["stats"] = result.stats
            except BaseException as err:
                err_holder["err"] = err
            finally:
                chunk_queue.put(SENTINEL)

        worker = threading.Thread(target=_worker, daemon=True, name="lunavox-stream")
        worker.start()

        # Keep a reference to the trampoline so the GC doesn't drop the
        # C-visible callback while synthesis is still running.
        _keep_alive = cb_trampoline  # noqa: F841

        # The terminal (is_last=True) chunk lands on the queue from
        # inside the C chunk callback, which fires BEFORE the worker
        # has had a chance to call _consume_audio and populate
        # stats_holder. If we yielded it immediately the caller
        # would see chunk.stats is None on the last chunk. Instead,
        # hold the terminal chunk, keep draining until SENTINEL
        # (which means the worker finished writing stats), then
        # yield the terminal chunk with the populated stats.
        held_terminal: Any = None
        try:
            while True:
                item = chunk_queue.get()
                if item is SENTINEL:
                    break
                audio, is_last = item
                if is_last:
                    held_terminal = audio
                    continue
                yield SynthesisChunk(
                    audio=audio,
                    sample_rate=sr_holder["rate"] or int(self.sample_rate),
                    is_last=False,
                    stats=None,
                )

            if held_terminal is not None:
                yield SynthesisChunk(
                    audio=held_terminal,
                    sample_rate=sr_holder["rate"] or int(self.sample_rate),
                    is_last=True,
                    stats=stats_holder["stats"],
                )
        finally:
            worker.join(timeout=60.0)
            if err_holder["err"] is not None:
                raise err_holder["err"]

    def _dispatch_stream(
        self,
        text: str,
        voice: Voice,
        cp: _capi.CParams,
        cb: Any,
    ) -> Any:
        """Route one streaming synthesis call to the right ``_streaming`` C symbol.

        Mirrors :meth:`_dispatch` exactly but for the streaming variant
        family. Adding a voice mode adds one branch here and one
        classmethod to :class:`Voice` — no other caller changes.
        """
        text_b = text.encode("utf-8")
        if voice.mode is SynthesisMode.BASE:
            return self._lib.lunavox_synthesize_streaming(
                self.handle, text_b, ctypes.byref(cp), cb, None
            )
        if voice.mode is SynthesisMode.CLONE_FILE:
            if voice.reference_path is None:
                raise LunavoxSynthesisError("Voice.clone_file requires reference_path")
            return self._lib.lunavox_synthesize_with_voice_file_streaming(
                self.handle,
                text_b,
                str(voice.reference_path).encode("utf-8"),
                ctypes.byref(cp),
                cb,
                None,
            )
        if voice.mode is SynthesisMode.CUSTOM:
            if not voice.speaker:
                raise LunavoxSynthesisError("Voice.custom requires speaker id")
            return self._lib.lunavox_synthesize_custom_streaming(
                self.handle,
                text_b,
                voice.speaker.encode("utf-8"),
                (voice.instruct or "").encode("utf-8"),
                ctypes.byref(cp),
                cb,
                None,
            )
        if voice.mode is SynthesisMode.DESIGN:
            if not (voice.instruct and voice.instruct.strip()):
                raise LunavoxSynthesisError("Voice.design requires non-empty instruct")
            return self._lib.lunavox_synthesize_design_streaming(
                self.handle,
                text_b,
                voice.instruct.encode("utf-8"),
                ctypes.byref(cp),
                cb,
                None,
            )
        raise LunavoxSynthesisError(f"Unsupported voice mode: {voice.mode}")
