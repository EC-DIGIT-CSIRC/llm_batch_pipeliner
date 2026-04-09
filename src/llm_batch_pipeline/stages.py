"""Stage wiring: connects all processing stages into a :class:`Pipeline`.

Each stage function receives a :class:`PipelineContext` and returns a
:class:`StageResult`.  Stages communicate via ``ctx.files``,
``ctx.filtered_files``, and ``ctx.artifacts``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console

from llm_batch_pipeline.backends.base import SubmissionResult
from llm_batch_pipeline.backends.ollama_backend import OllamaBackend
from llm_batch_pipeline.backends.openai_backend import OpenAIBackend
from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.evaluation import (
    EvalReport,
    PredictionRow,
    evaluate,
    infer_ground_truth_from_prefix,
    load_category_map,
    load_ground_truth_csv,
)
from llm_batch_pipeline.export import export_evaluation_xlsx, export_results_xlsx
from llm_batch_pipeline.filters import run_filter_chain
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.pipeline import Pipeline, PipelineContext, StageResult
from llm_batch_pipeline.plugins.registry import get_plugin
from llm_batch_pipeline.render import render_batch
from llm_batch_pipeline.transforms import run_transform_chain
from llm_batch_pipeline.tui import make_stage_progress, update_stage_speed
from llm_batch_pipeline.validation import validate_batch_output

logger = logging.getLogger("llm_batch_pipeline.stages")


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def stage_discover(ctx: PipelineContext) -> StageResult:
    """Stage 1: Discover and read input files."""
    plugin = get_plugin(ctx.config.plugin_name)
    reader = plugin.reader

    input_dir = ctx.config.input_dir
    if input_dir is None or not input_dir.is_dir():
        return StageResult(name="discover", status="failed", error=f"Input directory not found: {input_dir}")

    all_paths = sorted(p for p in input_dir.iterdir() if p.is_file())

    files = []
    progress, task_id = make_stage_progress("Discovering", len(all_paths), console=ctx.console)
    progress.start()
    try:
        for path in all_paths:
            if reader.can_read(path):
                parsed = reader.read(path)
                files.append(parsed)
            progress.update(task_id, advance=1)
            update_stage_speed(progress, task_id)
    finally:
        progress.stop()

    ctx.files = files
    log_event(logger, f"Discovered {len(files)} files", step="discover", status="ok", count=len(files))
    return StageResult(name="discover", status="completed", detail=f"{len(files)} files")


def stage_filter_1(ctx: PipelineContext) -> StageResult:
    """Stage 2a: Pre-transform filter chain."""
    plugin = get_plugin(ctx.config.plugin_name)
    total = len(ctx.files)
    progress, task_id = make_stage_progress("Filtering (pre)", total, console=ctx.console)

    def _advance() -> None:
        progress.update(task_id, advance=1)
        update_stage_speed(progress, task_id)

    progress.start()
    try:
        result = run_filter_chain(ctx.files, plugin.pre_filters, chain_name="filter_1", on_progress=_advance)
    finally:
        progress.stop()

    ctx.filtered_files = result.kept
    return StageResult(
        name="filter_1",
        status="completed",
        detail=f"kept {result.kept_count}/{result.total_input}",
        artifacts={"dropped": result.dropped},
    )


def stage_transform(ctx: PipelineContext) -> StageResult:
    """Stage 2b: Transform chain."""
    plugin = get_plugin(ctx.config.plugin_name)
    total = len(ctx.filtered_files)
    progress, task_id = make_stage_progress("Transforming", total, console=ctx.console)

    def _advance() -> None:
        progress.update(task_id, advance=1)
        update_stage_speed(progress, task_id)

    progress.start()
    try:
        ctx.filtered_files = run_transform_chain(
            ctx.filtered_files, plugin.transformers, chain_name="transform", on_progress=_advance
        )
    finally:
        progress.stop()

    return StageResult(name="transform", status="completed", detail=f"{len(ctx.filtered_files)} files")


def stage_filter_2(ctx: PipelineContext) -> StageResult:
    """Stage 2c: Post-transform filter chain."""
    plugin = get_plugin(ctx.config.plugin_name)
    total = len(ctx.filtered_files)
    progress, task_id = make_stage_progress("Filtering (post)", total, console=ctx.console)

    def _advance() -> None:
        progress.update(task_id, advance=1)
        update_stage_speed(progress, task_id)

    progress.start()
    try:
        result = run_filter_chain(ctx.filtered_files, plugin.post_filters, chain_name="filter_2", on_progress=_advance)
    finally:
        progress.stop()

    ctx.filtered_files = result.kept
    return StageResult(
        name="filter_2",
        status="completed",
        detail=f"kept {result.kept_count}/{result.total_input}",
    )


def stage_render(ctx: PipelineContext) -> StageResult:
    """Stage 3: Render batch JSONL files."""
    plugin = get_plugin(ctx.config.plugin_name)
    shard_paths = render_batch(ctx.filtered_files, plugin.reader, ctx.config)
    ctx.artifacts["shard_paths"] = [str(p) for p in shard_paths]
    return StageResult(
        name="render",
        status="completed",
        detail=f"{len(shard_paths)} shard(s), {len(ctx.filtered_files)} requests",
    )


def stage_review(ctx: PipelineContext) -> StageResult:
    """Stage 4: Human review of batch JSONL (optional)."""
    if ctx.config.auto_approve:
        return StageResult(name="review", status="completed", detail="auto-approved")

    shard_paths = [Path(p) for p in ctx.artifacts.get("shard_paths", [])]
    if not shard_paths:
        return StageResult(name="review", status="failed", error="No shard files to review")

    total_lines = 0
    total_bytes = 0
    for sp in shard_paths:
        content = sp.read_text(encoding="utf-8")
        total_lines += len([ln for ln in content.splitlines() if ln.strip()])
        total_bytes += sp.stat().st_size

    ctx.console.print(f"\n  Batch files: {len(shard_paths)}")
    ctx.console.print(f"  Total requests: {total_lines}")
    ctx.console.print(f"  Total size: {total_bytes / 1024:.1f} KB")
    ctx.console.print(f"  Location: [cyan]{shard_paths[0].parent}[/cyan]\n")

    # In non-interactive mode, auto-approve
    if not ctx.console.is_terminal:
        return StageResult(name="review", status="completed", detail="auto-approved (non-interactive)")

    try:
        from rich.prompt import Confirm  # noqa: PLC0415

        if not Confirm.ask("Proceed with submission?", default=True, console=ctx.console):
            return StageResult(name="review", status="failed", error="User rejected batch")
    except (EOFError, KeyboardInterrupt):
        return StageResult(name="review", status="failed", error="Review interrupted")

    return StageResult(name="review", status="completed", detail="approved")


def stage_submit(ctx: PipelineContext) -> StageResult:
    """Stage 5: Submit batch to backend."""
    shard_paths = [Path(p) for p in ctx.artifacts.get("shard_paths", [])]
    if not shard_paths:
        return StageResult(name="submit", status="failed", error="No shard files found")

    backend_name = ctx.config.backend
    if backend_name == "openai":
        backend = OpenAIBackend()
    elif backend_name == "ollama":
        backend = OllamaBackend()
    else:
        return StageResult(name="submit", status="failed", error=f"Unknown backend: {backend_name}")

    # Submit each shard (usually just one)
    all_results: list[SubmissionResult] = []
    for shard_path in shard_paths:
        result = backend.submit(
            shard_path,
            ctx.config,
            console=ctx.console,
            metrics=ctx.metrics,
        )
        all_results.append(result)

    # Aggregate
    total_completed = sum(r.completed_requests for r in all_results)
    total_failed = sum(r.failed_requests for r in all_results)

    # Store output file path for downstream stages
    output_files = [str(r.output_file) for r in all_results if r.output_file]
    ctx.artifacts["output_files"] = output_files
    ctx.artifacts["submission_results"] = [r.summary for r in all_results]

    status = "completed" if all(r.status in ("completed", "submitted") for r in all_results) else "completed"
    return StageResult(
        name="submit",
        status=status,
        detail=f"ok={total_completed} failed={total_failed}",
    )


def stage_validate(ctx: PipelineContext) -> StageResult:
    """Stage 7: Validate results against schema."""
    output_files = ctx.artifacts.get("output_files", [])
    if not output_files:
        return StageResult(name="validate", status="failed", error="No output files to validate")

    schema_path = ctx.config.schema_file
    if schema_path is None:
        candidate = ctx.config.batch_dir / "schema.py"
        if candidate.is_file():
            schema_path = candidate

    all_valid: list[dict] = []
    all_invalid: list[dict] = []
    for output_path in output_files:
        result = validate_batch_output(Path(output_path), schema_path)
        for row in result.valid_rows:
            entry = dict(row.parsed_data or {})
            entry["filename"] = row.custom_id
            all_valid.append(entry)
        for row in result.invalid_rows:
            all_invalid.append({"filename": row.custom_id, "error": row.error_message})

        ctx.metrics.record_validation(ctx.config.batch_name, "valid")

    ctx.artifacts["validated_rows"] = all_valid
    ctx.artifacts["validation_errors"] = all_invalid

    # Write validated results
    results_dir = ctx.config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "validated.json"
    results_path.write_text(json.dumps(all_valid, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return StageResult(
        name="validate",
        status="completed",
        detail=f"valid={len(all_valid)} invalid={len(all_invalid)}",
    )


def stage_output_transform(ctx: PipelineContext) -> StageResult:
    """Stage 8: Transform validated output rows."""
    plugin = get_plugin(ctx.config.plugin_name)
    rows = ctx.artifacts.get("validated_rows", [])

    if plugin.output_transformer and rows:
        rows = plugin.output_transformer.apply(rows)
        ctx.artifacts["validated_rows"] = rows

    return StageResult(name="output_transform", status="completed", detail=f"{len(rows)} rows")


def stage_evaluate(ctx: PipelineContext) -> StageResult:
    """Stage 9: Evaluate against ground truth."""
    rows = ctx.artifacts.get("validated_rows", [])
    if not rows:
        return StageResult(name="evaluate", status="skipped", detail="No validated rows")

    # Load ground truth
    gt_map: dict[str, str] = {}
    category_map: dict[str, str] = {}

    gt_csv = ctx.config.ground_truth_csv
    if gt_csv is None:
        candidate = ctx.config.evaluation_dir / "ground-truth.csv"
        if candidate.is_file():
            gt_csv = candidate

    cat_map_path = ctx.config.category_map_file
    if cat_map_path is None:
        candidate = ctx.config.evaluation_dir / "category-map.json"
        if candidate.is_file():
            cat_map_path = candidate

    if gt_csv and gt_csv.is_file():
        gt_map = load_ground_truth_csv(gt_csv)
    if cat_map_path and cat_map_path.is_file():
        category_map = load_category_map(cat_map_path)

    if not gt_map and not category_map:
        return StageResult(name="evaluate", status="skipped", detail="No ground truth available")

    # Determine label and confidence fields
    label_field = ctx.config.label_field or "label"
    confidence_field = ctx.config.confidence_field or "confidence"

    # Auto-detect from schema if not explicitly set
    if ctx.config.schema_file and not ctx.config.label_field:
        from llm_batch_pipeline.schema_loader import infer_confidence_field, infer_label_field  # noqa: PLC0415

        detected_label = infer_label_field(ctx.config.schema_file)
        detected_conf = infer_confidence_field(ctx.config.schema_file)
        if detected_label:
            label_field = detected_label
        if detected_conf:
            confidence_field = detected_conf

    # Build prediction rows
    predictions: list[PredictionRow] = []
    for row in rows:
        filename = row.get("filename", "")
        predicted = str(row.get(label_field, "")).strip().lower()
        confidence = row.get(confidence_field)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None

        # Resolve ground truth
        ground_truth = gt_map.get(filename)
        if ground_truth is None and category_map:
            ground_truth = infer_ground_truth_from_prefix(filename, category_map)

        if ground_truth is None:
            continue

        predictions.append(
            PredictionRow(
                custom_id=filename,
                ground_truth=ground_truth,
                predicted=predicted,
                confidence=confidence,
                raw_output=row,
            )
        )

    if not predictions:
        return StageResult(name="evaluate", status="skipped", detail="No matchable predictions")

    report = evaluate(predictions, positive_class=ctx.config.positive_class)
    ctx.artifacts["eval_report"] = report.to_dict()
    ctx.artifacts["_eval_report_obj"] = report

    # Print summary to console
    _print_eval_summary(report, ctx.console)

    return StageResult(
        name="evaluate",
        status="completed",
        detail=f"accuracy={report.accuracy:.4f} macro_f1={report.macro_f1:.4f}",
    )


def stage_export(ctx: PipelineContext) -> StageResult:
    """Stage 10: Export results and evaluation to XLSX."""
    export_dir = ctx.config.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    schema_path = ctx.config.schema_file
    if schema_path is None:
        candidate = ctx.config.batch_dir / "schema.py"
        if candidate.is_file():
            schema_path = candidate

    # Export results
    rows = ctx.artifacts.get("validated_rows", [])
    if rows:
        export_results_xlsx(rows, export_dir / "results.xlsx", schema_path=schema_path)

    # Export evaluation
    report = ctx.artifacts.get("_eval_report_obj")
    if report and isinstance(report, EvalReport):
        export_evaluation_xlsx(report, export_dir / "evaluation.xlsx", schema_path=schema_path)
        # Also write JSON
        eval_json_path = export_dir / "evaluation.json"
        eval_json_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    return StageResult(name="export", status="completed", detail=f"{export_dir}")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_pipeline(config: BatchConfig, console: Console | None = None) -> Pipeline:
    """Construct the full 10-stage pipeline."""
    con = console or Console()
    pipeline = Pipeline(config.batch_name or "batch_run", config.batch_dir, console=con)

    pipeline.add_stage("discover", stage_discover)
    pipeline.add_stage("filter_1", stage_filter_1)
    pipeline.add_stage("transform", stage_transform)
    pipeline.add_stage("filter_2", stage_filter_2)
    pipeline.add_stage("render", stage_render)
    pipeline.add_stage("review", stage_review, optional=True)
    pipeline.add_stage("submit", stage_submit, retries=1)
    pipeline.add_stage("validate", stage_validate)
    pipeline.add_stage("output_transform", stage_output_transform)
    pipeline.add_stage("evaluate", stage_evaluate, optional=True)
    pipeline.add_stage("export", stage_export, optional=True)

    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_eval_summary(report: EvalReport, console: Console) -> None:
    """Print evaluation results as a Rich table."""
    from rich.table import Table  # noqa: PLC0415

    # Metrics table
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Accuracy", f"{report.accuracy:.4f}")
    table.add_row("Macro F1", f"{report.macro_f1:.4f}")
    table.add_row("Macro Precision", f"{report.macro_precision:.4f}")
    table.add_row("Macro Recall", f"{report.macro_recall:.4f}")
    table.add_row("Total Predictions", str(len(report.rows)))
    console.print(table)

    # Per-class table
    class_table = Table(title="Per-Class Metrics")
    class_table.add_column("Class", style="bold")
    class_table.add_column("Precision")
    class_table.add_column("Recall")
    class_table.add_column("F1")
    class_table.add_column("Support")
    for cm in report.per_class:
        class_table.add_row(
            cm.label,
            f"{cm.precision:.4f}",
            f"{cm.recall:.4f}",
            f"{cm.f1:.4f}",
            str(cm.support),
        )
    console.print(class_table)

    # Confusion matrix
    if report.labels and report.confusion:
        cm_table = Table(title="Confusion Matrix")
        cm_table.add_column("")
        for label in report.labels:
            cm_table.add_column(f"Pred:{label}")
        for i, label in enumerate(report.labels):
            row_vals = [str(report.confusion[i][j]) for j in range(len(report.labels))]
            cm_table.add_row(f"True:{label}", *row_vals)
        console.print(cm_table)
