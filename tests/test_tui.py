"""Tests for tui.py — formatting helpers and display snapshots."""

import pytest

from llm_batch_pipeline.tui import (
    build_display_snapshot,
    format_duration,
    format_eta,
    format_speed,
    format_status_line,
    print_pipeline_summary,
)


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0) == "00:00:00"

    def test_seconds(self):
        assert format_duration(45) == "00:00:45"

    def test_minutes(self):
        assert format_duration(125) == "00:02:05"

    def test_hours(self):
        assert format_duration(3661) == "01:01:01"


class TestFormatSpeed:
    def test_none(self):
        assert format_speed(None) == "?it/s"

    def test_zero(self):
        assert format_speed(0) == "?it/s"

    def test_fast(self):
        result = format_speed(5.5)
        assert result == "5.50it/s"

    def test_slow(self):
        result = format_speed(0.5)
        assert result == "2.00s/it"


class TestFormatEta:
    def test_none(self):
        assert format_eta(None) == "ETA --:--:--"

    def test_with_seconds(self):
        assert format_eta(120) == "ETA 00:02:00"


class TestBuildDisplaySnapshot:
    def test_basic(self):
        snap = build_display_snapshot(
            batch_id="batch_001",
            status="processing",
            total_requests=100,
            completed_requests=50,
            failed_requests=5,
            started_monotonic=0.0,
            state_started_monotonic=0.0,
            now_monotonic=10.0,
        )
        assert snap.total_requests == 100
        assert snap.completed_requests == 50
        assert snap.failed_requests == 5
        assert snap.processed_requests == 55
        assert snap.remaining_requests == 45
        assert snap.percent_complete == pytest.approx(55.0)
        assert snap.elapsed_seconds == 10.0
        assert snap.speed_items_per_sec is not None
        assert snap.speed_items_per_sec == 5.5
        assert snap.eta_seconds is not None

    def test_zero_total(self):
        snap = build_display_snapshot(
            batch_id="x",
            status="idle",
            total_requests=0,
            completed_requests=0,
            failed_requests=0,
            started_monotonic=0.0,
            state_started_monotonic=0.0,
            now_monotonic=0.0,
        )
        assert snap.percent_complete == 0.0
        assert snap.speed_items_per_sec is None


class TestFormatStatusLine:
    def test_produces_string(self):
        snap = build_display_snapshot(
            batch_id="batch_001_test",
            status="processing",
            total_requests=10,
            completed_requests=5,
            failed_requests=0,
            started_monotonic=0.0,
            state_started_monotonic=0.0,
            now_monotonic=5.0,
        )
        line = format_status_line(snap)
        assert "processing" in line
        assert "5/10" in line


class TestPrintPipelineSummary:
    def test_renders_without_error(self):
        """Just verify it doesn't crash."""
        results = [
            {"name": "discover", "status": "completed", "duration_ms": 100.0, "detail": "10 files"},
            {"name": "filter_1", "status": "completed", "duration_ms": 50.0},
            {"name": "submit", "status": "failed", "duration_ms": 200.0, "error": "timeout"},
        ]
        from rich.console import Console

        console = Console(file=None, quiet=True)
        print_pipeline_summary(results, console=console)
