"""Timing, statistics, and Prometheus metrics integration.

Provides:

* :func:`timed` — context-manager for measuring durations with
  ``time.perf_counter_ns()``.
* :class:`MetricsCollector` — wraps ``prometheus-client`` counters,
  histograms, and gauges.  Optionally starts an HTTP server on a
  configurable port so Prometheus / Grafana can scrape ``/metrics``.
  When the server is disabled, metrics are still collected locally and
  written to ``metrics.json`` at the end of the run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextmanager
def timed() -> Generator[dict[str, float]]:
    """Context-manager that measures wall-clock time in milliseconds.

    Usage::

        with timed() as t:
            do_work()
        print(t["duration_ms"])
    """
    result: dict[str, float] = {}
    start = time.perf_counter_ns()
    try:
        yield result
    finally:
        result["duration_ms"] = (time.perf_counter_ns() - start) / 1_000_000


# ---------------------------------------------------------------------------
# Local statistics accumulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageStats:
    """Accumulated statistics for a single pipeline stage."""

    stage: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float("inf") else None,
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
        }


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Wraps prometheus-client with optional HTTP server for Grafana scraping.

    When *port* is ``None`` the HTTP server is **not** started, but metrics
    are still collected locally and can be written to a JSON file via
    :meth:`write_summary`.
    """

    def __init__(self, *, port: int | None = None) -> None:
        self._registry = CollectorRegistry()
        self._port = port
        self._local_stats: dict[str, StageStats] = {}

        # --- Prometheus instruments ----------------------------------------

        self.runs_total = Counter(
            "llm_batch_pipeline_runs_total",
            "Total pipeline runs",
            ["batch_name", "status"],
            registry=self._registry,
        )
        self.stage_duration_seconds = Histogram(
            "llm_batch_pipeline_stage_duration_seconds",
            "Duration of each pipeline stage",
            ["batch_name", "stage"],
            buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600),
            registry=self._registry,
        )
        self.active_requests = Gauge(
            "llm_batch_pipeline_active_requests",
            "Currently in-flight LLM requests",
            ["backend"],
            registry=self._registry,
        )
        self.requests_total = Counter(
            "llm_batch_pipeline_requests_total",
            "Total LLM requests",
            ["backend", "status"],
            registry=self._registry,
        )
        self.request_duration_seconds = Histogram(
            "llm_batch_pipeline_request_duration_seconds",
            "Duration of individual LLM requests",
            ["backend", "model"],
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            registry=self._registry,
        )
        self.validation_total = Counter(
            "llm_batch_pipeline_validation_total",
            "Schema validation results",
            ["batch_name", "status"],
            registry=self._registry,
        )

        if self._port is not None:
            start_http_server(self._port, registry=self._registry)

    # --- Recording helpers -------------------------------------------------

    def record_stage(self, batch_name: str, stage: str, duration_ms: float, status: str) -> None:
        """Record stage completion for both Prometheus and local stats."""
        self.stage_duration_seconds.labels(batch_name=batch_name, stage=stage).observe(duration_ms / 1000)
        if status in ("completed", "ok"):
            self.runs_total.labels(batch_name=batch_name, status="stage_ok").inc()

        if stage not in self._local_stats:
            self._local_stats[stage] = StageStats(stage=stage)
        self._local_stats[stage].record(duration_ms)

    def record_request(self, backend: str, model: str, duration_ms: float, status: str) -> None:
        self.request_duration_seconds.labels(backend=backend, model=model).observe(duration_ms / 1000)
        self.requests_total.labels(backend=backend, status=status).inc()

    def record_validation(self, batch_name: str, status: str) -> None:
        self.validation_total.labels(batch_name=batch_name, status=status).inc()

    def inc_active(self, backend: str) -> None:
        self.active_requests.labels(backend=backend).inc()

    def dec_active(self, backend: str) -> None:
        self.active_requests.labels(backend=backend).dec()

    # --- Summary -----------------------------------------------------------

    def get_local_stats(self) -> dict[str, StageStats]:
        return dict(self._local_stats)

    def write_summary(self, path: Path) -> None:
        """Write local stage statistics to a JSON file."""
        data = {name: stats.to_dict() for name, stats in self._local_stats.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Token estimation (crude but cheap)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)
