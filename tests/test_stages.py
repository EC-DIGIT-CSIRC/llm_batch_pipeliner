"""Integration tests for stage wiring and batch roundtrips."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.pipeline import PipelineContext
from llm_batch_pipeline.stages import (
    build_pipeline,
    stage_discover,
    stage_filter_1,
    stage_filter_2,
    stage_render,
    stage_review,
    stage_transform,
)


def _make_ctx(batch_dir: Path) -> PipelineContext:
    config = BatchConfig(
        batch_dir=batch_dir,
        plugin_name="spam_detection",
        model="gemma4:latest",
        prompt_file=batch_dir / "prompt.txt",
        schema_file=batch_dir / "schema.py",
        auto_approve=True,
    )
    return PipelineContext(
        batch_dir=batch_dir,
        config=config,
        console=Console(file=None, quiet=True),
        metrics=MetricsCollector(),
    )


def test_build_pipeline_stage_order(batch_roundtrip_fixture_dir: Path) -> None:
    ctx = _make_ctx(batch_roundtrip_fixture_dir)
    pipeline = build_pipeline(ctx.config, console=ctx.console)

    assert pipeline.stage_names == [
        "discover",
        "filter_1",
        "transform",
        "filter_2",
        "render",
        "review",
        "submit",
        "validate",
        "output_transform",
        "evaluate",
        "export",
    ]


def test_stage_chain_renders_expected_jsonl(copy_batch_roundtrip_fixture: Path) -> None:
    ctx = _make_ctx(copy_batch_roundtrip_fixture)

    discover = stage_discover(ctx)
    assert discover.status == "completed"
    assert len(ctx.files) == 2

    pre_filter = stage_filter_1(ctx)
    assert pre_filter.status == "completed"
    assert len(ctx.filtered_files) == 2

    transform = stage_transform(ctx)
    assert transform.status == "completed"

    post_filter = stage_filter_2(ctx)
    assert post_filter.status == "completed"

    render = stage_render(ctx)
    assert render.status == "completed"

    shard_paths = [Path(path) for path in ctx.artifacts["shard_paths"]]
    assert len(shard_paths) == 1

    records = [json.loads(line) for line in shard_paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2
    assert (
        records[0]["body"]["instructions"]
        == (copy_batch_roundtrip_fixture / "prompt.txt").read_text(encoding="utf-8").strip()
    )
    assert records[0]["body"]["text"]["format"]["type"] == "json_schema"
    assert (copy_batch_roundtrip_fixture / "job" / "batch.jsonl").is_symlink()


def test_review_auto_approves_when_configured(copy_batch_roundtrip_fixture: Path) -> None:
    ctx = _make_ctx(copy_batch_roundtrip_fixture)
    assert stage_discover(ctx).status == "completed"
    assert stage_filter_1(ctx).status == "completed"
    assert stage_transform(ctx).status == "completed"
    assert stage_filter_2(ctx).status == "completed"
    assert stage_render(ctx).status == "completed"

    review = stage_review(ctx)
    assert review.status == "completed"
    assert review.detail == "auto-approved"
