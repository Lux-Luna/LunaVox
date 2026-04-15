"""Unit tests for :mod:`lunavox.serve.metrics`.

Exercise the metric instruments without a real engine — we just
need to make sure the registry is constructed cleanly, gauges
update via ``snapshot_pool``, and ``render`` produces parseable
Prometheus text.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "prometheus_client",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)


def test_metrics_registry_isolated_per_instance():
    """Two LunavoxMetrics instances must not collide on metric names."""
    from lunavox.serve.metrics import LunavoxMetrics

    a = LunavoxMetrics()
    b = LunavoxMetrics()
    assert a.registry is not b.registry


def test_metrics_render_returns_prometheus_text():
    from lunavox.serve.metrics import LunavoxMetrics

    metrics = LunavoxMetrics()
    metrics.requests_total.labels(voice="base", status="success").inc()
    metrics.request_duration_seconds.labels(voice="base").observe(0.42)
    metrics.rtf.labels(voice="base").observe(0.18)

    body, content_type = metrics.render()
    text = body.decode("utf-8")

    assert content_type.startswith("text/plain")
    assert "lunavox_requests_total" in text
    assert "lunavox_request_duration_seconds" in text
    assert "lunavox_rtf" in text
    # Counter increment with labels lands in the exposition.
    assert 'voice="base"' in text
    assert 'status="success"' in text


def test_snapshot_pool_updates_gauges_from_batch():
    from lunavox.serve.metrics import LunavoxMetrics

    class FakeBatch:
        batch_size = 4
        idle_count = 2

    metrics = LunavoxMetrics()
    metrics.snapshot_pool(FakeBatch())  # type: ignore[arg-type]

    body, _ = metrics.render()
    text = body.decode("utf-8")
    # Prometheus exposition uses bare numbers without quotes.
    assert "lunavox_pool_size 4.0" in text
    assert "lunavox_pool_idle 2.0" in text


def test_render_with_batch_refreshes_first():
    from lunavox.serve.metrics import LunavoxMetrics

    class FakeBatch:
        batch_size = 8
        idle_count = 5

    metrics = LunavoxMetrics()
    body, _ = metrics.render(FakeBatch())  # type: ignore[arg-type]
    text = body.decode("utf-8")
    assert "lunavox_pool_size 8.0" in text
    assert "lunavox_pool_idle 5.0" in text
