"""Timing, statistics, and OpenTelemetry export helpers.

Provides:

* :func:`timed` — context-manager for measuring durations with
  ``time.perf_counter_ns()``.
* :class:`MetricsCollector` — keeps local stage statistics and, when
  OTLP endpoints are configured via standard ``OTEL_EXPORTER_OTLP_*``
  environment variables, exports metrics and logs to an OpenTelemetry
  Collector.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

_METRICS_ENDPOINT_ENV_VARS = frozenset(
    {
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    }
)
_LOGS_ENDPOINT_ENV_VARS = frozenset(
    {
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    }
)
_SCALAR_ATTR_TYPES = (bool, str, int, float)


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
# OTLP logging handler
# ---------------------------------------------------------------------------


class OtelLogHandler(logging.Handler):
    """Forward Python log records to an OpenTelemetry logger provider."""

    def __init__(self, logger_provider: LoggerProvider) -> None:
        super().__init__()
        self._logger_provider = logger_provider

    def emit(self, record: logging.LogRecord) -> None:
        try:
            otel_logger = self._logger_provider.get_logger(record.name)
            otel_logger.emit(
                timestamp=int(record.created * 1_000_000_000),
                observed_timestamp=time.time_ns(),
                severity_text=_severity_text(record.levelname),
                severity_number=_severity_number(record.levelno),
                body=self.format(record) if self.formatter else record.getMessage(),
                attributes=_log_attributes(record),
            )
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        with suppress(Exception):
            self._logger_provider.force_flush()

    def close(self) -> None:
        self.flush()
        super().close()


# ---------------------------------------------------------------------------
# Metrics + logs collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Track local stage stats and optionally export OTLP metrics/logs."""

    def __init__(self, *, port: int | None = None) -> None:
        del port
        self._local_stats: dict[str, StageStats] = {}

        self._meter_provider: MeterProvider | None = None
        self._logger_provider: LoggerProvider | None = None

        self._runs_total = None
        self._stage_duration_seconds = None
        self._active_requests = None
        self._requests_total = None
        self._request_duration_seconds = None
        self._validation_total = None

        resource = _build_resource()

        if _metrics_export_enabled():
            reader = PeriodicExportingMetricReader(OTLPMetricExporter())
            self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            meter = self._meter_provider.get_meter("llm_batch_pipeline")
            self._runs_total = meter.create_counter(
                "llm_batch_pipeline_runs_total",
                description="Total full pipeline runs",
            )
            self._stage_duration_seconds = meter.create_histogram(
                "llm_batch_pipeline_stage_duration_seconds",
                unit="s",
                description="Duration of each pipeline stage",
            )
            self._active_requests = meter.create_up_down_counter(
                "llm_batch_pipeline_active_requests",
                description="Currently in-flight LLM requests",
            )
            self._requests_total = meter.create_counter(
                "llm_batch_pipeline_requests_total",
                description="Total LLM requests",
            )
            self._request_duration_seconds = meter.create_histogram(
                "llm_batch_pipeline_request_duration_seconds",
                unit="s",
                description="Duration of individual LLM requests",
            )
            self._validation_total = meter.create_counter(
                "llm_batch_pipeline_validation_total",
                description="Schema validation result rows",
            )

        if _logs_export_enabled():
            self._logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
            self._logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))

    def build_log_handler(self) -> logging.Handler | None:
        """Return an OTLP log handler when log export is enabled."""
        if self._logger_provider is None:
            return None
        return OtelLogHandler(self._logger_provider)

    def record_run(self, command: str, status: str) -> None:
        """Record the lifecycle of a full pipeline run."""
        if self._runs_total is not None:
            self._runs_total.add(1, attributes={"command": command, "status": status})

    def record_stage(self, batch_name: str, stage: str, duration_ms: float, status: str) -> None:
        """Record stage completion for OTLP export and local stats."""
        del batch_name
        if self._stage_duration_seconds is not None:
            self._stage_duration_seconds.record(duration_ms / 1000, attributes={"stage": stage, "status": status})

        if stage not in self._local_stats:
            self._local_stats[stage] = StageStats(stage=stage)
        self._local_stats[stage].record(duration_ms)

    def record_request(self, backend: str, model: str, duration_ms: float, status: str) -> None:
        if self._request_duration_seconds is not None:
            self._request_duration_seconds.record(
                duration_ms / 1000,
                attributes={"backend": backend, "model": model, "status": status},
            )
        if self._requests_total is not None:
            self._requests_total.add(1, attributes={"backend": backend, "model": model, "status": status})

    def record_validation(self, batch_name: str, status: str, count: int = 1) -> None:
        del batch_name
        if self._validation_total is not None and count > 0:
            self._validation_total.add(count, attributes={"status": status})

    def inc_active(self, backend: str) -> None:
        if self._active_requests is not None:
            self._active_requests.add(1, attributes={"backend": backend})

    def dec_active(self, backend: str) -> None:
        if self._active_requests is not None:
            self._active_requests.add(-1, attributes={"backend": backend})

    def get_local_stats(self) -> dict[str, StageStats]:
        return dict(self._local_stats)

    def write_summary(self, path: Path) -> None:
        """Write local stage statistics to a JSON file."""
        data = {name: stats.to_dict() for name, stats in self._local_stats.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def shutdown(self) -> None:
        """Flush and shut down OTLP exporters."""
        if self._logger_provider is not None:
            with suppress(Exception):
                self._logger_provider.force_flush()
            with suppress(Exception):
                self._logger_provider.shutdown()
        if self._meter_provider is not None:
            with suppress(Exception):
                self._meter_provider.force_flush()
            with suppress(Exception):
                self._meter_provider.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "llm-batch-pipeline"),
            "service.version": _package_version(),
        }
    )


def _metrics_export_enabled() -> bool:
    return any(os.getenv(name) for name in _METRICS_ENDPOINT_ENV_VARS)


def _logs_export_enabled() -> bool:
    return any(os.getenv(name) for name in _LOGS_ENDPOINT_ENV_VARS)


def _package_version() -> str:
    try:
        return version("llm-batch-pipeline")
    except PackageNotFoundError:
        return "0.1.0"


def _severity_text(levelname: str) -> str:
    return {"WARNING": "WARN", "CRITICAL": "FATAL"}.get(levelname, levelname)


def _severity_number(levelno: int) -> SeverityNumber:
    if levelno >= logging.CRITICAL:
        return SeverityNumber.FATAL
    if levelno >= logging.ERROR:
        return SeverityNumber.ERROR
    if levelno >= logging.WARNING:
        return SeverityNumber.WARN
    if levelno >= logging.INFO:
        return SeverityNumber.INFO
    return SeverityNumber.DEBUG


def _log_attributes(record: logging.LogRecord) -> dict[str, Any]:
    attrs: dict[str, Any] = {"logger.name": record.name}
    event = getattr(record, "event", None)
    if isinstance(event, dict):
        for key, value in event.items():
            if value is None:
                continue
            attrs[key] = _coerce_attr_value(value)
    return attrs


def _coerce_attr_value(value: Any) -> Any:
    if isinstance(value, _SCALAR_ATTR_TYPES):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, _SCALAR_ATTR_TYPES) for item in value):
        return list(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Token estimation (crude but cheap)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)
