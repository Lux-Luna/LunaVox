"""Smoke test for the LunaVox-trimmed vendored converter.

Phase 2 cut ~9,000 lines of upstream llama.cpp code from
``src/lunavox/model/conversion/hf_export/convert_hf_to_gguf.py``. The risk
surface is (a) the file no longer imports, (b) the ``@ModelBase.register``
decorators silently stop firing, or (c) dispatch for the one architecture
LunaVox actually emits (``Qwen3ForCausalLM``) breaks.

This test keeps the guarantee honest without requiring a real Qwen
checkpoint — it imports the module, inspects the class registry, and
exercises the architecture lookup. No GPU, no HF download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HF_EXPORT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "lunavox"
    / "model"
    / "conversion"
    / "hf_export"
)


pytest.importorskip(
    "torch",
    reason="converter requires torch (install [convert] extra)",
    exc_type=ImportError,
)
pytest.importorskip(
    "transformers",
    reason="converter requires transformers (install [convert] extra)",
    exc_type=ImportError,
)


@pytest.fixture(scope="module")
def converter():
    """Import the vendored converter with its sibling ``gguf-py`` path
    wired up, matching how ``convert_talker_predictor_llama.py`` loads
    it at runtime."""
    sys.path.insert(0, str(HF_EXPORT))
    try:
        import convert_hf_to_gguf  # type: ignore

        return convert_hf_to_gguf
    finally:
        # Leave the path in place so other tests in the same process see
        # the import; removing it would make the module unreloadable.
        pass


def test_module_imports(converter):
    assert converter is not None
    assert hasattr(converter, "ModelBase")
    assert hasattr(converter, "TextModel")
    assert hasattr(converter, "Qwen2Model")
    assert hasattr(converter, "Qwen3Model")
    assert hasattr(converter, "LazyTorchTensor")
    assert callable(converter.main)


def test_registry_contains_qwen3(converter):
    registry = converter.ModelBase._model_classes[converter.ModelType.TEXT]
    assert "Qwen3ForCausalLM" in registry, (
        "Qwen3ForCausalLM missing from dispatch — @ModelBase.register decorators "
        "may have been stripped along with the class body."
    )
    assert "Qwen2ForCausalLM" in registry


def test_registry_has_no_deleted_architectures(converter):
    """The trim removed every non-Qwen architecture class. Confirm a
    representative set of deleted arches no longer resolves, so a stale
    upstream sync that accidentally re-introduces them would fail
    loudly."""
    registry = converter.ModelBase._model_classes[converter.ModelType.TEXT]
    for deleted in ("LlamaForCausalLM", "MistralForCausalLM", "BertModel", "GemmaForCausalLM"):
        assert deleted not in registry, f"deleted architecture {deleted} is back"


def test_dispatch_qwen3_returns_qwen3model(converter):
    cls = converter.ModelBase.from_model_architecture("Qwen3ForCausalLM", converter.ModelType.TEXT)
    assert cls is converter.Qwen3Model


def test_dispatch_unknown_raises_not_implemented(converter):
    with pytest.raises(NotImplementedError):
        converter.ModelBase.from_model_architecture("LlamaForCausalLM", converter.ModelType.TEXT)


def test_qwen3_inherits_qwen2(converter):
    """The Qwen3 -> Qwen2 -> TextModel -> ModelBase inheritance chain is
    what makes set_vocab / prepare_tensors / gguf output correct. Cut a
    parent by accident and the child silently changes behavior."""
    assert issubclass(converter.Qwen3Model, converter.Qwen2Model)
    assert issubclass(converter.Qwen2Model, converter.TextModel)
    assert issubclass(converter.TextModel, converter.ModelBase)


def test_qwenmodel_static_helpers_survive(converter):
    """``TextModel._set_vocab_qwen`` calls into ``QwenModel.bpe`` and
    ``QwenModel.token_bytes_to_string`` for tiktoken-style vocabularies.
    The helpers must still be callable after the trim."""
    assert hasattr(converter, "QwenModel")
    assert callable(converter.QwenModel.token_bytes_to_string)
    assert callable(converter.QwenModel.bpe)
    # Exercise bpe on a trivial token pair so we notice if the
    # implementation ever drifts.
    result = converter.QwenModel.bpe({b"ab": 1}, b"ab")
    assert result == [b"ab"]


def test_mistral_format_is_rejected(converter, tmp_path, monkeypatch):
    """The Mistral-format code path is no longer supported. Confirm
    ``main()`` refuses rather than crashing on a missing class."""
    # Create a synthetic "model directory" so main() reaches the
    # is_mistral_format check.
    dir_model = tmp_path / "fake_model"
    dir_model.mkdir()
    (dir_model / "config.json").write_text("{}", encoding="utf-8")

    argv = [
        "convert_hf_to_gguf.py",
        str(dir_model),
        "--outfile",
        str(tmp_path / "out.gguf"),
        "--outtype",
        "f16",
        "--mistral-format",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(NotImplementedError, match="mistral-format"):
        converter.main()
