"""Tests for logging_utils.py — formatters and event logging."""

import json
import logging

from llm_batch_pipeline.logging_utils import (
    ConsoleFormatter,
    JsonFormatter,
    get_logger,
    log_event,
)


class TestJsonFormatter:
    def test_basic_format(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert data["message"] == "hello"
        assert data["level"] == "info"
        assert "timestamp" in data

    def test_with_event_data(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="parsed file",
            args=(),
            exc_info=None,
        )
        record.event = {"file_id": "test.eml", "step": "parse", "status": "ok"}  # type: ignore[attr-defined]
        result = formatter.format(record)
        data = json.loads(result)
        assert data["file_id"] == "test.eml"
        assert data["step"] == "parse"


class TestConsoleFormatter:
    def test_basic_format(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.event = {"source_filename": "file.eml", "step": "parse", "status": "ok"}  # type: ignore[attr-defined]
        result = formatter.format(record)
        assert "file.eml" in result
        assert "parse" in result
        assert "ok" in result

    def test_with_duration(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="done",
            args=(),
            exc_info=None,
        )
        record.event = {"step": "render", "status": "ok", "duration_ms": 42.5}  # type: ignore[attr-defined]
        result = formatter.format(record)
        assert "42.5ms" in result


class TestLogEvent:
    def test_emits_structured_event(self):
        test_logger = get_logger("test_log_event")
        handler = logging.handlers.MemoryHandler(capacity=10)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        log_event(test_logger, "test message", step="test", status="ok", duration_ms=10.0)

        assert len(handler.buffer) == 1
        record = handler.buffer[0]
        assert hasattr(record, "event")
        assert record.event["step"] == "test"  # type: ignore[attr-defined]

        test_logger.removeHandler(handler)


class TestGetLogger:
    def test_returns_namespaced_logger(self):
        logger = get_logger("mymodule")
        assert logger.name == "llm_batch_pipeline.mymodule"

    def test_base_logger(self):
        logger = get_logger()
        assert logger.name == "llm_batch_pipeline"
