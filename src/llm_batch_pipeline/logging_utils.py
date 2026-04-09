"""Queue-based structured JSONL logging.

Provides multiprocess-safe logging via :class:`~logging.handlers.QueueHandler`
and :class:`~logging.handlers.QueueListener`.  Two formatters run in the main
process:

* **JsonFormatter** — writes machine-readable JSONL to ``pipeline.jsonl``.
* **ConsoleFormatter** — writes compact human-readable lines to *stderr*.

Worker processes only push :class:`~logging.LogRecord` objects onto a shared
queue; the listener drains and dispatches them, preventing interleaved output.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any

LOGGER_NAME: str = "llm_batch_pipeline"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Serialize each log record to a single JSON line.

    Structured fields are passed via ``extra={"event": {...}}``.
    """

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = dict(getattr(record, "event", {}))
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(event)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact single-line format for terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = getattr(record, "event", {})
        source = event.get("source_filename") or event.get("file_id") or event.get("batch_name") or "-"
        step = event.get("step") or event.get("stage") or "log"
        status = event.get("status") or record.levelname.lower()
        duration = event.get("duration_ms")
        dur_frag = f" {duration:.1f}ms" if isinstance(duration, int | float) else ""
        return f"{source} {step} {status}{dur_frag}: {record.getMessage()}"


class ConsoleLogFilter(logging.Filter):
    """Suppress per-item log events from the console handler.

    Per-item events carry a ``file_id`` or ``custom_id`` in their
    structured ``event`` dict.  Summary and pipeline-level events do not
    and are allowed through.  Warnings and errors always pass regardless.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        event: dict[str, Any] = getattr(record, "event", {})
        return not (event.get("file_id") or event.get("custom_id"))


# ---------------------------------------------------------------------------
# Runtime container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoggingRuntime:
    """Holds all resources needed for clean teardown."""

    manager: Any  # multiprocessing.SyncManager
    queue: Any  # multiprocessing.Queue
    listener: QueueListener
    handlers: tuple[logging.Handler, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_logging(logs_dir: Path, *, level: str = "INFO") -> LoggingRuntime:
    """Set up queue-based logging and return a :class:`LoggingRuntime`.

    Call :func:`stop_logging` when the pipeline finishes.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    manager = multiprocessing.Manager()
    queue = manager.Queue(-1)

    pipeline_handler = logging.FileHandler(
        logs_dir / "pipeline.jsonl",
        mode="w",
        encoding="utf-8",
    )
    pipeline_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleLogFilter())

    listener = QueueListener(queue, pipeline_handler, console_handler, respect_handler_level=True)
    listener.start()

    _configure_root_logger(queue, level)

    return LoggingRuntime(
        manager=manager,
        queue=queue,
        listener=listener,
        handlers=(pipeline_handler, console_handler),
    )


def stop_logging(runtime: LoggingRuntime) -> None:
    """Flush and tear down all logging resources."""
    runtime.listener.stop()
    for handler in runtime.handlers:
        handler.close()
    runtime.manager.shutdown()


def configure_worker_logging(queue: Any, *, level: str = "INFO") -> None:
    """Configure logging in a worker process to push records to *queue*."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(QueueHandler(queue))
    logger.propagate = False


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    **event: Any,
) -> None:
    """Emit a structured log event.

    Example::

        log_event(logger, "Parsed file",
                  file_id="invoice.eml", step="parse", status="ok",
                  duration_ms=12.3)
    """
    logger.log(level, message, extra={"event": event})


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the pipeline namespace."""
    base = LOGGER_NAME
    return logging.getLogger(f"{base}.{name}" if name else base)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _configure_root_logger(queue: Any, level: str) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(QueueHandler(queue))
    logger.propagate = False
