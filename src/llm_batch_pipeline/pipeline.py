"""Lightweight pipeline orchestrator.

Provides a linear stage runner with:

* Named stages with configurable retries and optional flag.
* Per-stage timing via ``time.perf_counter_ns()``.
* Resume from a specific stage (``start_from``).
* Dry-run mode: prints stage plan without executing.
* Rich progress panel showing current stage and elapsed time.
* Pipeline state serialised to ``pipeline_state.json`` for diagnostics.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.tui import PipelineProgressReporter, print_pipeline_summary, print_stage_header

logger = logging.getLogger("llm_batch_pipeline.pipeline")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageResult:
    """Outcome of a single pipeline stage."""

    name: str
    status: Literal["completed", "failed", "skipped"] = "completed"
    duration_ms: float = 0.0
    error: str | None = None
    detail: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Remove empty artifacts for cleaner output
        if not d.get("artifacts"):
            d.pop("artifacts", None)
        return d


@dataclass
class PipelineContext:
    """Shared mutable context passed through all stages.

    Stages read/write ``files``, ``filtered_files``, and ``artifacts``
    to communicate data downstream.
    """

    batch_dir: Path
    config: BatchConfig
    console: Console
    metrics: MetricsCollector
    # Populated by stages
    files: list[Any] = field(default_factory=list)
    filtered_files: list[Any] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage descriptor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StageDef:
    name: str
    fn: Callable[[PipelineContext], StageResult]
    retries: int = 0
    optional: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Lightweight linear pipeline runner.

    Usage::

        pipeline = Pipeline("my_run", batch_dir, console=console)
        pipeline.add_stage("discover", discover_fn)
        pipeline.add_stage("filter_1", filter_fn)
        pipeline.add_stage("submit", submit_fn, retries=2)
        pipeline.add_stage("evaluate", eval_fn, optional=True)
        results = pipeline.run(ctx)
    """

    def __init__(
        self,
        name: str,
        batch_dir: Path,
        *,
        console: Console | None = None,
    ) -> None:
        self.name = name
        self.batch_dir = batch_dir
        self.console: Console = console or Console()
        self._stages: list[_StageDef] = []

    # -- builder API ---------------------------------------------------------

    def add_stage(
        self,
        name: str,
        fn: Callable[[PipelineContext], StageResult],
        *,
        retries: int = 0,
        optional: bool = False,
    ) -> Pipeline:
        """Append a stage.  Returns ``self`` for chaining."""
        self._stages.append(_StageDef(name=name, fn=fn, retries=retries, optional=optional))
        return self

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    # -- execution -----------------------------------------------------------

    def run(
        self,
        ctx: PipelineContext,
        *,
        start_from: str | None = None,
        dry_run: bool = False,
    ) -> list[StageResult]:
        """Execute the pipeline and return results for every stage."""
        results: list[StageResult] = []
        skipping = start_from is not None
        progress = PipelineProgressReporter(len(self._stages), console=self.console)

        if dry_run:
            self._print_dry_run()
            return results

        ctx.metrics.record_run("run", "started", backend=ctx.config.backend)

        log_event(
            logger,
            f"Pipeline '{self.name}' starting",
            step="pipeline",
            status="starting",
            backend=ctx.config.backend,
            stages=len(self._stages),
            start_from=start_from or "(beginning)",
        )

        progress.start()
        pipeline_start = time.perf_counter_ns()

        try:
            for idx, stage_def in enumerate(self._stages, 1):
                if skipping:
                    if stage_def.name == start_from:
                        skipping = False
                    else:
                        results.append(StageResult(name=stage_def.name, status="skipped"))
                        progress.advance(f"{stage_def.name} (skipped)")
                        continue

                print_stage_header(stage_def.name, idx, len(self._stages), console=self.console)
                result = self._run_stage(stage_def, ctx)
                results.append(result)
                progress.advance(stage_def.name)

                # Record metrics
                ctx.metrics.record_stage(
                    batch_name=ctx.config.batch_name,
                    stage=stage_def.name,
                    duration_ms=result.duration_ms,
                    status=result.status,
                    backend=ctx.config.backend,
                )

                if result.status == "failed" and not stage_def.optional:
                    log_event(
                        logger,
                        f"Pipeline aborted at stage '{stage_def.name}'",
                        step="pipeline",
                        status="aborted",
                        backend=ctx.config.backend,
                        failed_stage=stage_def.name,
                        error=result.error,
                        level=logging.ERROR,
                    )
                    break
        finally:
            progress.stop()

        total_ms = (time.perf_counter_ns() - pipeline_start) / 1_000_000
        all_ok = all(r.status in ("completed", "skipped") for r in results)
        ctx.metrics.record_run("run", "completed" if all_ok else "failed", backend=ctx.config.backend)

        log_event(
            logger,
            f"Pipeline '{self.name}' finished",
            step="pipeline",
            status="completed" if all_ok else "partial",
            backend=ctx.config.backend,
            duration_ms=total_ms,
            stages_completed=sum(1 for r in results if r.status == "completed"),
            stages_failed=sum(1 for r in results if r.status == "failed"),
            stages_skipped=sum(1 for r in results if r.status == "skipped"),
        )

        # Persist state and show summary
        self._save_state(results, total_ms)
        print_pipeline_summary([r.to_dict() for r in results], console=self.console)

        return results

    # -- internal ------------------------------------------------------------

    def _run_stage(self, stage_def: _StageDef, ctx: PipelineContext) -> StageResult:
        """Execute a single stage with retry logic."""
        last_error: str | None = None
        attempts = stage_def.retries + 1

        for attempt in range(1, attempts + 1):
            start_ns = time.perf_counter_ns()
            try:
                log_event(
                    logger,
                    f"Stage '{stage_def.name}' starting (attempt {attempt}/{attempts})",
                    step=stage_def.name,
                    status="starting",
                    backend=ctx.config.backend,
                    attempt=attempt,
                )
                result = stage_def.fn(ctx)
                result.duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

                if result.status == "completed":
                    log_event(
                        logger,
                        f"Stage '{stage_def.name}' completed",
                        step=stage_def.name,
                        status="completed",
                        backend=ctx.config.backend,
                        duration_ms=result.duration_ms,
                    )
                    return result
                last_error = result.error

            except Exception as exc:  # noqa: BLE001
                duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                last_error = f"{exc.__class__.__name__}: {exc}"
                log_event(
                    logger,
                    f"Stage '{stage_def.name}' raised: {last_error}",
                    step=stage_def.name,
                    status="error",
                    backend=ctx.config.backend,
                    duration_ms=duration_ms,
                    attempt=attempt,
                    error=last_error,
                    level=logging.ERROR,
                )

            if attempt < attempts:
                log_event(
                    logger,
                    f"Retrying stage '{stage_def.name}'",
                    step=stage_def.name,
                    status="retrying",
                    backend=ctx.config.backend,
                    attempt=attempt + 1,
                )

        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000  # type: ignore[possibly-undefined]
        return StageResult(
            name=stage_def.name,
            status="failed",
            duration_ms=duration_ms,
            error=last_error,
        )

    def _print_dry_run(self) -> None:
        self.console.print("\n[bold]Dry-run mode — stages that would execute:[/bold]\n")
        for idx, stage_def in enumerate(self._stages, 1):
            opt = " [dim](optional)[/dim]" if stage_def.optional else ""
            retry = f" [dim](retries: {stage_def.retries})[/dim]" if stage_def.retries else ""
            self.console.print(f"  {idx:2d}. {stage_def.name}{opt}{retry}")
        self.console.print()

    def _save_state(self, results: list[StageResult], total_ms: float) -> None:
        state_path = self.batch_dir / "pipeline_state.json"
        state = {
            "pipeline": self.name,
            "total_duration_ms": round(total_ms, 3),
            "stages": [r.to_dict() for r in results],
        }
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        except OSError:
            logger.warning("Could not save pipeline state to %s", state_path)
