"""Tests for Ollama backend helpers."""

from llm_batch_pipeline.backends.ollama_backend import OllamaExecutionResult, _summarise_row_results


class TestSummariseRowResults:
    def test_summarises_wall_clock_row_stats(self):
        results = [
            OllamaExecutionResult(index=0, custom_id="a", success_record={"ok": True}, duration_ms=100.0),
            OllamaExecutionResult(index=1, custom_id="b", error_record={"error": True}, duration_ms=250.0),
            OllamaExecutionResult(index=2, custom_id="c", success_record={"ok": True}, duration_ms=400.0),
        ]

        summary = _summarise_row_results(results)

        assert summary == {
            "rows_total": 3,
            "rows_success": 2,
            "rows_failed": 1,
            "row_duration_avg_ms": 250.0,
            "row_duration_p50_ms": 250.0,
            "row_duration_min_ms": 100.0,
            "row_duration_max_ms": 400.0,
        }

    def test_handles_empty_result_list(self):
        summary = _summarise_row_results([])

        assert summary == {
            "rows_total": 0,
            "rows_success": 0,
            "rows_failed": 0,
            "row_duration_avg_ms": 0.0,
            "row_duration_p50_ms": 0.0,
            "row_duration_min_ms": 0.0,
            "row_duration_max_ms": 0.0,
        }
