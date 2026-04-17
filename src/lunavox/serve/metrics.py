"""Prometheus metrics for ``lunavox serve``.

Phase 5C wires four core metrics into the FastAPI handlers so an
ops team can scrape ``/metrics`` and monitor a deployment:

* ``lunavox_pool_size`` (gauge) — total engines configured
* ``lunavox_pool_idle`` (gauge) — currently idle engines
* ``lunavox_requests_total`` (counter) — labelled by voice mode
  (``base`` / ``clone`` / ``custom`` / ``design``) and ``status``
  (``success`` / ``error``)
* ``lunavox_request_duration_seconds`` (histogram) — server-side
  latency per request, labelled by voice mode
* ``lunavox_rtf`` (histogram) — engine real-time factor per request
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from lunavox.runtime import BatchEngine


# Latency / RTF histogram buckets — chosen so a typical RTX 3090
# Vulkan run (RTF ~0.15, latency ~1.3s for a 25-word sentence)
# lands comfortably in the middle of both ranges.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
_RTF_BUCKETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


class LunavoxMetrics:
    """Owns one ``CollectorRegistry`` and the metric instruments.

    Kept as a class (rather than module-level metrics) so multiple
    apps in the same process — useful in tests — can each have an
    isolated registry without `Duplicated timeseries` errors from
    prometheus-client's default global registry.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.pool_size = Gauge(
            "lunavox_pool_size",
            "Total engines configured in the BatchEngine pool.",
            registry=self.registry,
        )
        self.pool_idle = Gauge(
            "lunavox_pool_idle",
            "Engines currently idle (free for new requests).",
            registry=self.registry,
        )
        self.requests_total = Counter(
            "lunavox_requests_total",
            "Total synthesis requests served, labelled by voice mode and status.",
            labelnames=("voice", "status"),
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "lunavox_request_duration_seconds",
            "Server-side wall time per synthesis request, labelled by voice mode.",
            labelnames=("voice",),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.rtf = Histogram(
            "lunavox_rtf",
            "Real-time factor reported by the engine, labelled by voice mode.",
            labelnames=("voice",),
            buckets=_RTF_BUCKETS,
            registry=self.registry,
        )

    def snapshot_pool(self, batch: BatchEngine) -> None:
        """Refresh the gauge metrics from the live :class:`BatchEngine`.

        Called inline by handlers before they block on the pool, and
        also opportunistically by ``/metrics`` so a scrape with no
        traffic still reports current state.
        """
        self.pool_size.set(batch.batch_size)
        self.pool_idle.set(batch.idle_count)

    def render(self, batch: BatchEngine | None = None) -> tuple[bytes, str]:
        """Render the registry to Prometheus text format.

        Returns ``(body_bytes, content_type)`` so the FastAPI handler
        can stamp the right ``Content-Type`` header without importing
        prometheus_client itself. ``batch`` is optional — when
        provided we refresh the pool gauges first so the metric
        snapshot matches the moment of the scrape.
        """
        if batch is not None:
            self.snapshot_pool(batch)
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
