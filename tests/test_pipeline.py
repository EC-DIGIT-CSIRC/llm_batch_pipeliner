"""Tests for pipeline.py — pipeline orchestrator."""

from pathlib import Path

from rich.console import Console

from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.pipeline import Pipeline, PipelineContext, StageResult


def _make_ctx(tmp_path: Path) -> PipelineContext:
    cfg = BatchConfig(batch_dir=tmp_path)
    return PipelineContext(
        batch_dir=tmp_path,
        config=cfg,
        console=Console(file=None, quiet=True),
        metrics=MetricsCollector(),
    )


def _ok_stage(ctx: PipelineContext) -> StageResult:
    return StageResult(name="ok_stage", status="completed", detail="all good")


def _fail_stage(ctx: PipelineContext) -> StageResult:
    return StageResult(name="fail_stage", status="failed", error="something broke")


def _counter_stage(ctx: PipelineContext) -> StageResult:
    count = ctx.artifacts.get("counter", 0) + 1
    ctx.artifacts["counter"] = count
    return StageResult(name="counter", status="completed", detail=f"count={count}")


class TestPipeline:
    def test_single_stage(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("step1", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert len(results) == 1
        assert results[0].status == "completed"

    def test_multiple_stages(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        pipeline.add_stage("b", _ok_stage)
        pipeline.add_stage("c", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

    def test_failure_stops_pipeline(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        pipeline.add_stage("b", _fail_stage)
        pipeline.add_stage("c", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert len(results) == 2  # Stops at failure
        assert results[1].status == "failed"

    def test_optional_failure_continues(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        pipeline.add_stage("b", _fail_stage, optional=True)
        pipeline.add_stage("c", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert len(results) == 3
        assert results[1].status == "failed"
        assert results[2].status == "completed"

    def test_start_from_skips_stages(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _counter_stage)
        pipeline.add_stage("b", _counter_stage)
        pipeline.add_stage("c", _counter_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx, start_from="b")
        assert results[0].status == "skipped"
        assert results[1].status == "completed"
        assert results[2].status == "completed"

    def test_dry_run(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        pipeline.add_stage("b", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx, dry_run=True)
        assert results == []  # No stages executed

    def test_context_artifacts_persist(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _counter_stage)
        pipeline.add_stage("b", _counter_stage)
        ctx = _make_ctx(tmp_path)
        pipeline.run(ctx)
        assert ctx.artifacts["counter"] == 2

    def test_saves_pipeline_state(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        ctx = _make_ctx(tmp_path)
        pipeline.run(ctx)
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.is_file()

    def test_stage_names(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("discover", _ok_stage)
        pipeline.add_stage("filter", _ok_stage)
        pipeline.add_stage("render", _ok_stage)
        assert pipeline.stage_names == ["discover", "filter", "render"]

    def test_duration_recorded(self, tmp_path):
        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _ok_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert results[0].duration_ms >= 0

    def test_exception_in_stage_handled(self, tmp_path):
        def _raising_stage(ctx: PipelineContext) -> StageResult:
            msg = "kaboom"
            raise RuntimeError(msg)

        pipeline = Pipeline("test", tmp_path, console=Console(file=None, quiet=True))
        pipeline.add_stage("a", _raising_stage)
        ctx = _make_ctx(tmp_path)
        results = pipeline.run(ctx)
        assert results[0].status == "failed"
        assert "kaboom" in (results[0].error or "")
