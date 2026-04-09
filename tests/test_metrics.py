"""Tests for metrics.py — timing, StageStats, MetricsCollector."""

import time

from llm_batch_pipeline.metrics import (
    MetricsCollector,
    StageStats,
    estimate_tokens,
    timed,
)


class TestTimed:
    def test_measures_duration(self):
        with timed() as t:
            time.sleep(0.01)
        assert "duration_ms" in t
        assert t["duration_ms"] >= 5  # At least 5ms (sleep 10ms with margin)

    def test_records_on_exception(self):
        try:
            with timed() as t:
                msg = "test"
                raise ValueError(msg)
        except ValueError:
            pass
        assert "duration_ms" in t


class TestStageStats:
    def test_single_record(self):
        s = StageStats(stage="discover")
        s.record(100.0)
        assert s.count == 1
        assert s.total_ms == 100.0
        assert s.min_ms == 100.0
        assert s.max_ms == 100.0
        assert s.mean_ms == 100.0

    def test_multiple_records(self):
        s = StageStats(stage="filter")
        s.record(50.0)
        s.record(100.0)
        s.record(150.0)
        assert s.count == 3
        assert s.total_ms == 300.0
        assert s.min_ms == 50.0
        assert s.max_ms == 150.0
        assert s.mean_ms == 100.0

    def test_to_dict(self):
        s = StageStats(stage="render")
        s.record(42.5)
        d = s.to_dict()
        assert d["stage"] == "render"
        assert d["count"] == 1
        assert d["total_ms"] == 42.5

    def test_empty_mean(self):
        s = StageStats(stage="empty")
        assert s.mean_ms == 0.0


class TestMetricsCollector:
    def test_record_stage(self):
        mc = MetricsCollector()
        mc.record_stage("batch1", "discover", 100.0, "completed")
        stats = mc.get_local_stats()
        assert "discover" in stats
        assert stats["discover"].count == 1

    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request("ollama", "llama3", 500.0, "ok")
        # Just verify no exception

    def test_write_summary(self, tmp_path):
        mc = MetricsCollector()
        mc.record_stage("batch1", "render", 50.0, "completed")
        mc.record_stage("batch1", "submit", 200.0, "completed")
        summary_path = tmp_path / "metrics.json"
        mc.write_summary(summary_path)
        assert summary_path.is_file()

        import json

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "render" in data
        assert "submit" in data

    def test_active_requests(self):
        mc = MetricsCollector()
        mc.inc_active("ollama")
        mc.dec_active("ollama")
        # Just verify no exception


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") >= 1

    def test_empty(self):
        assert estimate_tokens("") == 1

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100
