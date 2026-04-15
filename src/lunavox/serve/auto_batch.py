"""VRAM-aware automatic ``--batch-size`` resolution.

When the user passes ``lunavox serve --batch-size auto`` we probe
the GPU's free VRAM via ``pynvml`` (already shipped with most CUDA
installs and bundled by torch) and divide it by an estimated
per-engine footprint to pick a safe pool size. Failures degrade
gracefully: any error in the probe path returns ``DEFAULT_FALLBACK``
with a logged reason, so the server still starts.

Per-engine footprint depends on the model size. For Qwen3-TTS the
empirical numbers (talker + predictor KV caches + ONNX decoder
state) are roughly:

* 0.6B base: ~1.0 GB / engine
* 1.7B base: ~3.0 GB / engine

We don't have the model loaded yet at the moment we need to pick
the batch size, so we read it off the model directory name as a
heuristic ('base_small' / 'custom_small' → 0.6B, anything else
→ 1.7B). The :data:`PER_SLOT_OVERRIDE_ENV` env var lets ops force
a specific value when the heuristic is wrong.

Result is always clamped to ``[MIN_BATCH, MAX_BATCH]`` so a tiny
GPU still gets ``batch_size=1`` and a huge GPU doesn't get an
unbounded number of slots.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)

#: Env var override for the per-slot VRAM estimate, in megabytes.
#: Set to ``LUNAVOX_VRAM_PER_SLOT_MB=2048`` to assume 2 GB per engine.
PER_SLOT_OVERRIDE_ENV = "LUNAVOX_VRAM_PER_SLOT_MB"

#: Hard bounds on the auto-resolved batch size.
MIN_BATCH = 1
MAX_BATCH = 16

#: What we fall back to when the probe fails.
DEFAULT_FALLBACK = 4

#: Default per-slot footprint, megabytes.
_DEFAULT_PER_SLOT_MB_SMALL = 1100
_DEFAULT_PER_SLOT_MB_LARGE = 3100


def _per_slot_mb_for_model(model_dir: Path) -> int:
    """Pick a reasonable per-slot VRAM estimate for ``model_dir``.

    Honors ``LUNAVOX_VRAM_PER_SLOT_MB`` first; falls back to a
    rule-of-thumb based on the model directory name.
    """
    override = os.environ.get(PER_SLOT_OVERRIDE_ENV, "").strip()
    if override:
        try:
            return max(64, int(override))
        except ValueError:
            LOG.warning(
                "Ignoring invalid %s=%r (expected positive integer MB).",
                PER_SLOT_OVERRIDE_ENV,
                override,
            )

    name = model_dir.name.lower()
    if "small" in name:
        return _DEFAULT_PER_SLOT_MB_SMALL
    return _DEFAULT_PER_SLOT_MB_LARGE


def _probe_free_vram_mb() -> int | None:
    """Return free VRAM on the active GPU in megabytes, or ``None``.

    Tries ``pynvml`` (the lib torch already pulls in for CUDA users)
    and gives up on any error — including "no NVIDIA GPU at all" on
    AMD / Intel / CPU-only hosts. The caller falls back to the
    default in that case.
    """
    # `pynvml` has been renamed to `nvidia-ml-py` and emits a
    # FutureWarning on import from the legacy name. Silence it so
    # startup doesn't spam the ops console on every `serve` run.
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import pynvml  # type: ignore[import-not-found]
    except ImportError:
        LOG.info("pynvml unavailable; --batch-size auto cannot probe VRAM.")
        return None

    try:
        pynvml.nvmlInit()
    except Exception as err:  # pragma: no cover — depends on host hardware
        LOG.info("nvmlInit failed (%s); --batch-size auto falling back.", err)
        return None

    try:
        # Read the first GPU only; multi-GPU deployments should pin
        # via CUDA_VISIBLE_DEVICES so device 0 is the right one.
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free // (1024 * 1024))
    except Exception as err:
        LOG.warning("nvmlDeviceGetMemoryInfo failed: %s", err)
        return None
    finally:
        # Shutdown failures are non-fatal — we only care about the
        # snapshot we already collected.
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def resolve_batch_size(
    requested: str | int,
    *,
    model_dir: Path,
) -> int:
    """Resolve the user-supplied ``--batch-size`` flag to a concrete int.

    ``requested`` is whatever the CLI received: either a positive
    integer (use as-is, clamped to the bounds) or the literal
    string ``"auto"`` (probe + compute).
    """
    if isinstance(requested, int):
        return max(MIN_BATCH, min(MAX_BATCH, requested))

    if requested != "auto":
        try:
            return max(MIN_BATCH, min(MAX_BATCH, int(requested)))
        except ValueError as err:
            raise ValueError(
                f"--batch-size must be an integer or 'auto', got {requested!r}"
            ) from err

    free_mb = _probe_free_vram_mb()
    if free_mb is None:
        LOG.info(
            "--batch-size auto: VRAM probe unavailable, using fallback %d.",
            DEFAULT_FALLBACK,
        )
        return DEFAULT_FALLBACK

    per_slot_mb = _per_slot_mb_for_model(model_dir)
    # Reserve 20% headroom so a long sentence with extra KV growth
    # doesn't push us over the edge.
    usable_mb = int(free_mb * 0.8)
    raw_slots = max(1, usable_mb // per_slot_mb)
    clamped = max(MIN_BATCH, min(MAX_BATCH, raw_slots))
    LOG.info(
        "--batch-size auto: free=%d MB, per_slot=%d MB, headroom=80%%, picked=%d",
        free_mb,
        per_slot_mb,
        clamped,
    )
    return clamped
