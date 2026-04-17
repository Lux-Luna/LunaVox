"""ctypes struct-layout sanity checks for ``lunavox.runtime._capi``.

These tests import the module without triggering ``load_library``, so
they run on every host (no liblunavox required). They verify the
Python mirror of the C ABI stays in lock-step with
``src/lunavox_c_api.h`` — the first line of defence against silent
struct-field drift.
"""

from __future__ import annotations

import ctypes

from lunavox.runtime import _capi
from lunavox.runtime.params import SynthesisMode, default_params


def test_cparams_field_order_and_types():
    """Order and types must match ``LunavoxSynthesisParams`` in the C
    header; reordering would silently corrupt every synth call."""
    expected = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
        ("ref_text", ctypes.c_char_p),
    ]
    assert _capi.CParams._fields_ == expected


def test_cmemstats_field_order_and_types():
    """Mirrors ``LunavoxMemStats`` in the C header. Field drift silently
    misaligns uint64 reads on strict-alignment targets, so enforce both
    name order and ctypes types."""
    expected = [
        ("rss_start_bytes", ctypes.c_uint64),
        ("rss_end_bytes", ctypes.c_uint64),
        ("rss_peak_bytes", ctypes.c_uint64),
        ("vram_start_bytes", ctypes.c_uint64),
        ("vram_end_bytes", ctypes.c_uint64),
        ("vram_peak_bytes", ctypes.c_uint64),
        ("vram_measured", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]
    actual = _capi.CMemStats._fields_
    assert [f[0] for f in actual] == [f[0] for f in expected]
    assert [f[1] for f in actual] == [f[1] for f in expected]
    # Size must match the C struct: 6×8 + 2×4 = 56 bytes.
    assert ctypes.sizeof(_capi.CMemStats) == 56


def test_caudio_field_order_and_types():
    expected = [
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
        ("_pad", ctypes.c_float),
        ("mem", _capi.CMemStats),
    ]
    actual = _capi.CAudio._fields_
    assert [f[0] for f in actual] == [f[0] for f in expected]
    assert [f[1] for f in actual] == [f[1] for f in expected]


def test_synthesis_params_defaults_match_c_defaults():
    """``default_params`` is the Python contract for the C
    ``lunavox_default_params`` struct — keep them in sync."""
    p = default_params()
    assert p.temperature == 0.6
    assert p.top_p == 1.0
    assert p.top_k == 50
    assert p.n_threads == 4
    assert p.language_id == -1
    assert p.ref_text is None


def test_synthesis_mode_values_are_stable():
    """Enum values leak through logs and telemetry — changing them is
    a breaking change for downstream dashboards."""
    assert SynthesisMode.BASE.value == "base"
    assert SynthesisMode.CLONE_FILE.value == "clone_file"
    assert SynthesisMode.CUSTOM.value == "custom"
    assert SynthesisMode.DESIGN.value == "design"


def test_log_callback_prototype_shape():
    """Signature must be (int level, char* message, void* user)."""
    proto = _capi.LOG_CALLBACK_T
    assert proto is not None
    null_cb = proto(0)
    assert null_cb is not None
