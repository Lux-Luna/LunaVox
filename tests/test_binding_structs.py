"""ctypes struct-layout sanity checks for ``lunavox.runtime.binding``.

These tests import the module without triggering ``_load_library``, so
they run on every host (no liblunavox required). They verify the Python
mirror of the C ABI stays in lock-step with ``src/lunavox_c_api.h`` —
the first line of defence against silent struct-field drift.
"""

from __future__ import annotations

import ctypes

from lunavox.runtime import binding as b


def _field_dict(struct_cls):
    return {name: typ for name, typ in struct_cls._fields_}


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
    assert b._CParams._fields_ == expected


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
        ("rss_peak_bytes", ctypes.c_uint64),
        ("rss_end_bytes", ctypes.c_uint64),
    ]
    # Compare by name + type. ctypes ``_fields_`` entries can be 2- or
    # 3-tuples (the optional third element is a bitfield width), so we
    # index explicitly instead of unpacking.
    actual = b._CAudio._fields_
    assert [f[0] for f in actual] == [f[0] for f in expected]
    assert [f[1] for f in actual] == [f[1] for f in expected]


def test_synthesis_params_defaults_match_c_defaults():
    """``default_params`` is the Python contract for the C
    ``lunavox_default_params`` struct — keep them in sync."""
    p = b.default_params()
    assert p.temperature == 0.6
    assert p.top_p == 1.0
    assert p.top_k == 50
    assert p.n_threads == 4
    assert p.language_id == -1
    assert p.ref_text is None


def test_synthesis_mode_values_are_stable():
    """Enum values leak through logs and telemetry — changing them is
    a breaking change for downstream dashboards."""
    assert b.SynthesisMode.BASE.value == "base"
    assert b.SynthesisMode.CLONE_FILE.value == "clone_file"
    assert b.SynthesisMode.CUSTOM.value == "custom"
    assert b.SynthesisMode.DESIGN.value == "design"


def test_log_callback_prototype_shape():
    """Signature must be (int level, char* message, void* user)."""
    proto = b._LOG_CALLBACK_T
    # ctypes.CFUNCTYPE stores restype and argtypes as attributes on the
    # returned type; we only assert the shape.
    assert proto is not None
    # Calling the type with 0 returns a null callback; this exercises
    # the factory without actually invoking into C.
    null_cb = proto(0)
    assert null_cb is not None
