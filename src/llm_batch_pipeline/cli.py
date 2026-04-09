"""Command-line interface for llm-batch-pipeline.

Subcommands
-----------
init       Create a new batch job directory with scaffolding.
run        Execute the full pipeline (discover → export).
render     Render batch JSONL from input files.
submit     Submit rendered JSONL to an LLM backend.
validate   Validate LLM output against a Pydantic schema.
evaluate   Evaluate validated results against ground truth.
export     Export results and evaluation to XLSX / JSON.
list       List registered plugins.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from llm_batch_pipeline.config import (
    BatchConfig,
    apply_toml_overrides,
    create_batch_dir,
    load_batch_toml,
    resolve_batch_dir,
)
from llm_batch_pipeline.logging_utils import start_logging, stop_logging
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.pipeline import PipelineContext
from llm_batch_pipeline.plugins.registry import list_plugins

logger = logging.getLogger("llm_batch_pipeline.cli")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="llm-batch-pipeline",
        description="Generic LLM batch processing pipeline.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    _add_init_parser(subparsers)
    _add_run_parser(subparsers)
    _add_render_parser(subparsers)
    _add_submit_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_export_parser(subparsers)
    _add_list_parser(subparsers)

    return parser


# -- shared argument groups --------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared across most subcommands."""
    parser.add_argument(
        "--batch-dir",
        dest="batch_dir",
        type=Path,
        help="Batch job directory (auto-resolved if name given)",
    )
    parser.add_argument(
        "--batch-jobs-root",
        dest="batch_jobs_root",
        type=Path,
        default=Path("batches"),
        help="Root directory for batch jobs (default: ./batches)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--metrics-port",
        dest="metrics_port",
        type=int,
        default=None,
        help="Port for Prometheus metrics HTTP server (disabled by default)",
    )


def _add_plugin_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plugin",
        dest="plugin_name",
        required=True,
        help="Plugin name (use 'list' subcommand to see available plugins)",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", dest="model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--prompt-file", dest="prompt_file", type=Path, help="Path to prompt text file")
    parser.add_argument("--schema-file", dest="schema_file", type=Path, help="Path to Pydantic schema .py file")


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        dest="backend",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM backend (default: openai)",
    )
    parser.add_argument(
        "--base-url",
        dest="base_urls",
        action="append",
        help="Ollama base URL (can repeat for multi-server sharding)",
    )
    parser.add_argument("--num-shards", dest="num_shards", type=int, help="Number of Ollama shards")
    parser.add_argument(
        "--num-parallel-jobs",
        dest="num_parallel_jobs",
        type=int,
        default=3,
        help="Parallel jobs per Ollama shard (default: 3)",
    )
    parser.add_argument(
        "--request-timeout",
        dest="request_timeout_seconds",
        type=int,
        default=600,
        help="Per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--poll-interval",
        dest="poll_interval_seconds",
        type=int,
        default=15,
        help="OpenAI batch poll interval in seconds (default: 15)",
    )
    parser.add_argument("--completion-window", dest="completion_window", default="24h")
    parser.add_argument(
        "-k",
        "--insecure",
        dest="insecure",
        action="store_true",
        default=False,
        help="Disable TLS certificate verification",
    )
    parser.add_argument("--no-wait", dest="no_wait", action="store_true", help="Submit and return without waiting")

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-override", dest="prompt_override", help="Inline prompt override at submit time")
    prompt_group.add_argument(
        "--prompt-override-file",
        dest="prompt_override_file",
        type=Path,
        help="File containing prompt override text",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ground-truth-csv", dest="ground_truth_csv", type=Path, help="CSV with (filename, label)")
    parser.add_argument("--category-map", dest="category_map_file", type=Path, help="JSON prefix→label mapping")
    parser.add_argument("--label-field", dest="label_field", help="Field name for predicted label")
    parser.add_argument("--confidence-field", dest="confidence_field", help="Field name for confidence score")
    parser.add_argument("--positive-class", dest="positive_class", help="Positive class for binary metrics")


# -- subcommand parsers ------------------------------------------------------


def _add_init_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("init", help="Create a new batch job directory")
    p.add_argument("name", help="Batch job name (becomes directory suffix)")
    p.add_argument(
        "--batch-jobs-root",
        dest="batch_jobs_root",
        type=Path,
        default=Path("batches"),
        help="Root directory for batch jobs (default: ./batches)",
    )
    _add_plugin_arg(p)
    _add_model_args(p)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("run", help="Execute the full pipeline")
    _add_common_args(p)
    _add_plugin_arg(p)
    _add_model_args(p)
    _add_backend_args(p)
    _add_eval_args(p)
    p.add_argument("--input-dir", dest="input_dir", type=Path, help="Input files directory")
    p.add_argument("--auto-approve", dest="auto_approve", action="store_true", help="Skip human review")
    p.add_argument("--start-from", dest="start_from", help="Resume from a specific stage")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Show plan without executing")
    p.add_argument(
        "--max-requests-per-shard",
        dest="max_requests_per_shard",
        type=int,
        default=50_000,
    )
    p.add_argument(
        "--max-bytes-per-shard",
        dest="max_bytes_per_shard",
        type=int,
        default=190 * 1024 * 1024,
    )


def _add_render_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("render", help="Render batch JSONL from input files")
    _add_common_args(p)
    _add_plugin_arg(p)
    _add_model_args(p)
    p.add_argument("--input-dir", dest="input_dir", type=Path, help="Input files directory")
    p.add_argument("--max-requests-per-shard", dest="max_requests_per_shard", type=int, default=50_000)
    p.add_argument(
        "--max-bytes-per-shard",
        dest="max_bytes_per_shard",
        type=int,
        default=190 * 1024 * 1024,
    )


def _add_submit_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("submit", help="Submit rendered JSONL to an LLM backend")
    _add_common_args(p)
    _add_backend_args(p)
    _add_model_args(p)
    p.add_argument("--batch-jsonl", dest="batch_jsonl", type=Path, help="Explicit JSONL file to submit")
    p.add_argument("--resume-batch-id", dest="resume_batch_id", help="Resume monitoring an existing OpenAI batch")


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("validate", help="Validate LLM output against schema")
    _add_common_args(p)
    p.add_argument("--schema-file", dest="schema_file", type=Path, help="Pydantic schema .py file")
    p.add_argument("--output-jsonl", dest="output_jsonl", type=Path, help="LLM output JSONL file to validate")


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("evaluate", help="Evaluate results against ground truth")
    _add_common_args(p)
    _add_eval_args(p)
    p.add_argument("--schema-file", dest="schema_file", type=Path, help="Pydantic schema .py file")


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("export", help="Export results and evaluation to XLSX/JSON")
    _add_common_args(p)
    p.add_argument("--schema-file", dest="schema_file", type=Path, help="Pydantic schema .py file")


def _add_list_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    subparsers.add_parser("list", help="List registered plugins")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> BatchConfig:
    """Build a :class:`BatchConfig` from parsed CLI arguments.

    Loads ``config.toml`` from the batch directory when present, then
    overlays CLI arguments (CLI wins).
    """
    batch_dir = getattr(args, "batch_dir", None)
    batch_jobs_root = getattr(args, "batch_jobs_root", Path("batches"))

    # Resolve batch_dir
    batch_dir = resolve_batch_dir(batch_jobs_root, str(batch_dir)) if batch_dir is not None else Path(".")

    config = BatchConfig(batch_dir=batch_dir)

    # Load TOML overrides from batch dir
    toml_path = batch_dir / "config.toml"
    toml_data = load_batch_toml(toml_path)
    if toml_data:
        apply_toml_overrides(config, toml_data)

    # Apply CLI arguments (CLI wins over TOML)
    _apply_cli_overrides(config, args)

    return config


def _apply_cli_overrides(config: BatchConfig, args: argparse.Namespace) -> None:
    """Overlay CLI arguments onto config, skipping None values."""
    mappings = {
        "plugin_name": "plugin_name",
        "model": "model",
        "input_dir": "input_dir",
        "prompt_file": "prompt_file",
        "schema_file": "schema_file",
        "backend": "backend",
        "num_shards": "num_shards",
        "num_parallel_jobs": "num_parallel_jobs",
        "request_timeout_seconds": "request_timeout_seconds",
        "poll_interval_seconds": "poll_interval_seconds",
        "completion_window": "completion_window",
        "insecure": "insecure",
        "no_wait": "no_wait",
        "prompt_override": "prompt_override",
        "prompt_override_file": "prompt_override_file",
        "ground_truth_csv": "ground_truth_csv",
        "category_map_file": "category_map_file",
        "label_field": "label_field",
        "confidence_field": "confidence_field",
        "positive_class": "positive_class",
        "metrics_port": "metrics_port",
        "log_level": "log_level",
        "auto_approve": "auto_approve",
        "start_from": "start_from",
        "dry_run": "dry_run",
        "max_requests_per_shard": "max_requests_per_shard",
        "max_bytes_per_shard": "max_bytes_per_shard",
        "resume_batch_id": "resume_batch_id",
    }

    for cli_attr, config_attr in mappings.items():
        value = getattr(args, cli_attr, None)
        if value is not None:
            setattr(config, config_attr, value)

    # Handle base_urls list (--base-url repeatable flag)
    base_urls = getattr(args, "base_urls", None)
    if base_urls:
        config.base_urls = base_urls

    # Derive batch_name from batch_dir stem if not set
    if not config.batch_name:
        config.batch_name = config.batch_dir.name


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Handle ``init`` subcommand."""
    batch_jobs_root = args.batch_jobs_root
    name = args.name

    console = Console()
    try:
        batch_dir = create_batch_dir(batch_jobs_root, name)
    except FileExistsError:
        console.print(f"[red]Batch directory already exists for name: {name}[/red]")
        return 1

    # Write a config.toml with initial settings
    toml_lines = [
        f'plugin_name = "{args.plugin_name}"',
        f'model = "{args.model}"',
    ]
    if args.prompt_file:
        toml_lines.append(f'prompt_file = "{args.prompt_file}"')
    if args.schema_file:
        toml_lines.append(f'schema_file = "{args.schema_file}"')

    (batch_dir / "config.toml").write_text("\n".join(toml_lines) + "\n", encoding="utf-8")

    console.print(f"Created batch directory: [cyan]{batch_dir}[/cyan]")
    console.print("  input/       — place input files here")
    console.print("  evaluation/  — place ground-truth files here")
    console.print("  config.toml  — batch configuration")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``run`` subcommand: execute the full pipeline."""
    from llm_batch_pipeline.stages import build_pipeline  # noqa: PLC0415

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        pipeline = build_pipeline(config, console=console)
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )
        results = pipeline.run(ctx, start_from=config.start_from, dry_run=config.dry_run)

        # Write metrics summary
        metrics.write_summary(config.logs_dir / "metrics.json")

        # Return non-zero if any required stage failed
        if any(r.status == "failed" for r in results):
            return 1
        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_render(args: argparse.Namespace) -> int:
    """Handle ``render`` subcommand: discover + filter + transform + render."""
    from llm_batch_pipeline.stages import (  # noqa: PLC0415
        stage_discover,
        stage_filter_1,
        stage_filter_2,
        stage_render,
        stage_transform,
    )

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )

        for stage_fn in [stage_discover, stage_filter_1, stage_transform, stage_filter_2, stage_render]:
            result = stage_fn(ctx)
            console.print(f"  {result.name}: {result.status} — {result.detail or ''}")
            if result.status == "failed":
                console.print(f"  [red]Error: {result.error}[/red]")
                return 1

        shard_paths = ctx.artifacts.get("shard_paths", [])
        console.print(f"\nRendered {len(shard_paths)} shard(s) to [cyan]{config.job_dir}[/cyan]")
        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_submit(args: argparse.Namespace) -> int:
    """Handle ``submit`` subcommand."""
    from llm_batch_pipeline.stages import stage_submit  # noqa: PLC0415

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )

        # Determine shard paths
        batch_jsonl = getattr(args, "batch_jsonl", None)
        if batch_jsonl:
            ctx.artifacts["shard_paths"] = [str(batch_jsonl)]
        else:
            # Auto-discover shards from job dir
            job_dir = config.job_dir
            if job_dir.is_dir():
                shards = sorted(job_dir.glob("batch-*.jsonl"))
                if shards:
                    ctx.artifacts["shard_paths"] = [str(s) for s in shards]

        if not ctx.artifacts.get("shard_paths"):
            console.print("[red]No batch JSONL files found. Run 'render' first or specify --batch-jsonl.[/red]")
            return 1

        result = stage_submit(ctx)
        console.print(f"  submit: {result.status} — {result.detail or ''}")
        if result.status == "failed":
            console.print(f"  [red]Error: {result.error}[/red]")
            return 1

        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle ``validate`` subcommand."""
    from llm_batch_pipeline.stages import stage_validate  # noqa: PLC0415

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )

        # Determine output files
        output_jsonl = getattr(args, "output_jsonl", None)
        if output_jsonl:
            ctx.artifacts["output_files"] = [str(output_jsonl)]
        else:
            # Auto-discover from output dir
            output_dir = config.output_dir
            if output_dir.is_dir():
                output_files = sorted(output_dir.glob("*.jsonl"))
                if output_files:
                    ctx.artifacts["output_files"] = [str(f) for f in output_files]

        if not ctx.artifacts.get("output_files"):
            console.print("[red]No output JSONL files found. Run 'submit' first or specify --output-jsonl.[/red]")
            return 1

        # Override schema_file if given on CLI
        schema_file = getattr(args, "schema_file", None)
        if schema_file:
            config.schema_file = schema_file

        result = stage_validate(ctx)
        console.print(f"  validate: {result.status} — {result.detail or ''}")
        if result.status == "failed":
            console.print(f"  [red]Error: {result.error}[/red]")
            return 1

        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Handle ``evaluate`` subcommand."""
    import json as json_mod  # noqa: PLC0415

    from llm_batch_pipeline.stages import stage_evaluate  # noqa: PLC0415

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )

        # Load validated results
        results_path = config.results_dir / "validated.json"
        if not results_path.is_file():
            console.print("[red]No validated results found. Run 'validate' first.[/red]")
            return 1

        ctx.artifacts["validated_rows"] = json_mod.loads(results_path.read_text(encoding="utf-8"))

        result = stage_evaluate(ctx)
        console.print(f"  evaluate: {result.status} — {result.detail or ''}")

        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_export(args: argparse.Namespace) -> int:
    """Handle ``export`` subcommand."""
    import json as json_mod  # noqa: PLC0415

    from llm_batch_pipeline.evaluation import EvalReport  # noqa: PLC0415
    from llm_batch_pipeline.stages import stage_export  # noqa: PLC0415

    config = _build_config(args)
    console = Console()
    metrics = MetricsCollector(port=config.metrics_port)

    logging_runtime = start_logging(config.logs_dir, level=config.log_level)
    try:
        ctx = PipelineContext(
            batch_dir=config.batch_dir,
            config=config,
            console=console,
            metrics=metrics,
        )

        # Load validated results if available
        results_path = config.results_dir / "validated.json"
        if results_path.is_file():
            ctx.artifacts["validated_rows"] = json_mod.loads(results_path.read_text(encoding="utf-8"))

        # Load evaluation report if available
        eval_path = config.export_dir / "evaluation.json"
        if eval_path.is_file():
            eval_data = json_mod.loads(eval_path.read_text(encoding="utf-8"))
            ctx.artifacts["eval_report"] = eval_data
            ctx.artifacts["_eval_report_obj"] = EvalReport.from_dict(eval_data)

        # Override schema_file if given
        schema_file = getattr(args, "schema_file", None)
        if schema_file:
            config.schema_file = schema_file

        result = stage_export(ctx)
        console.print(f"  export: {result.status} — {result.detail or ''}")

        return 0
    finally:
        stop_logging(logging_runtime)


def _cmd_list(_args: argparse.Namespace) -> int:
    """Handle ``list`` subcommand."""
    console = Console()
    plugins = list_plugins()
    if not plugins:
        console.print("No plugins registered.")
        return 0

    console.print("[bold]Registered plugins:[/bold]")
    for name in plugins:
        console.print(f"  - {name}")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "init": _cmd_init,
    "run": _cmd_run,
    "render": _cmd_render,
    "submit": _cmd_submit,
    "validate": _cmd_validate,
    "evaluate": _cmd_evaluate,
    "export": _cmd_export,
    "list": _cmd_list,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the llm-batch-pipeline CLI."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help()
        return 0

    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2

    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as exc:
        console = Console()
        console.print(f"[red]{exc}[/red]")
        return 1
    except KeyError as exc:
        console = Console()
        console.print(f"[red]{exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
