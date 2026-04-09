"""Rich TUI components: progress bars, status reporting, summary tables.

Core abstraction: :class:`StatusReporter` protocol defines the three-method
lifecycle (``start`` / ``update`` / ``stop``).  The concrete
:class:`RichStatusReporter` renders animated progress bars in interactive
terminals and falls back to static one-line prints when piped.

Colour conventions:
    - ``[red]`` errors
    - ``[yellow]`` warnings / in-progress
    - ``[green]`` success
    - ``[cyan]`` file paths
    - ``[bold]`` emphasis
    - ``[dim]`` inactive / unset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Display data contract
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchDisplaySnapshot:
    """Frozen view of batch-processing progress.

    Computed by :func:`build_display_snapshot`; consumed by
    :class:`StatusReporter` implementations.
    """

    batch_id: str
    status: str
    total_requests: int
    completed_requests: int
    failed_requests: int
    processed_requests: int
    remaining_requests: int
    percent_complete: float
    elapsed_seconds: float
    state_elapsed_seconds: float
    speed_items_per_sec: float | None
    eta_seconds: float | None


# ---------------------------------------------------------------------------
# StatusReporter protocol
# ---------------------------------------------------------------------------


class StatusReporter(Protocol):
    """Three-method lifecycle for pluggable batch progress display."""

    def start(self, snapshot: BatchDisplaySnapshot) -> None: ...
    def update(self, snapshot: BatchDisplaySnapshot) -> None: ...
    def stop(self, snapshot: BatchDisplaySnapshot) -> None: ...


# ---------------------------------------------------------------------------
# Rich implementation
# ---------------------------------------------------------------------------


class RichStatusReporter:
    """Animated Rich progress bar with terminal-detection fallback."""

    def __init__(self, console: Console | None = None) -> None:
        self.console: Console = console or Console()
        self._live = self.console.is_terminal
        self._progress: Progress | None = None
        self._task_id: int | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, snapshot: BatchDisplaySnapshot) -> None:
        if self._print_static(snapshot):
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[status]}[/bold]"),
            TextColumn("{task.fields[batch_suffix]}"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("ok={task.fields[ok]} fail={task.fields[failed]} rem={task.fields[remaining]}"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[speed]}"),
            TextColumn("{task.fields[eta]}"),
            console=self.console,
            transient=True,
        )
        fields = _progress_fields(snapshot)
        total = fields.pop("total")
        completed = fields.pop("completed")
        self._progress.start()
        self._task_id = self._progress.add_task(  # pylint: disable=repeated-keyword
            "batch",
            total=total,
            completed=completed,
            **fields,
        )

    def update(self, snapshot: BatchDisplaySnapshot) -> None:
        if self._print_static(snapshot):
            return
        if self._progress is not None and self._task_id is not None:
            fields = _progress_fields(snapshot)
            total = fields.pop("total")
            completed = fields.pop("completed")
            self._progress.update(  # pylint: disable=repeated-keyword
                self._task_id,
                total=total,
                completed=completed,
                **fields,
            )

    def stop(self, snapshot: BatchDisplaySnapshot) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._print_static(snapshot)

    # -- fallback ------------------------------------------------------------

    def _print_static(self, snapshot: BatchDisplaySnapshot) -> bool:
        if self._live and self._progress is not None:
            return False
        if not self._live:
            self.console.print(format_status_line(snapshot))
            return True
        return False


# ---------------------------------------------------------------------------
# Pipeline stage progress panel
# ---------------------------------------------------------------------------


class PipelineProgressReporter:
    """Shows pipeline stage progress as a Rich panel with a progress bar."""

    def __init__(self, total_stages: int, console: Console | None = None) -> None:
        self.console: Console = console or Console()
        self._total = total_stages
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def start(self) -> None:
        if not self.console.is_terminal:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[stage_name]}[/bold]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("pipeline", total=self._total, completed=0, stage_name="starting")

    def advance(self, stage_name: str) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1, stage_name=stage_name)
        elif not self.console.is_terminal:
            self.console.print(f"  Stage: {stage_name}")

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    total = int(max(0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_speed(speed: float | None) -> str:
    """Format speed as ``X.XXit/s`` or ``X.XXs/it``."""
    if speed is None or speed <= 0:
        return "?it/s"
    if speed >= 1.0:
        return f"{speed:.2f}it/s"
    return f"{1.0 / speed:.2f}s/it"


def format_eta(eta_seconds: float | None) -> str:
    """Format ETA as ``ETA HH:MM:SS`` or ``ETA --:--:--``."""
    if eta_seconds is None:
        return "ETA --:--:--"
    return f"ETA {format_duration(eta_seconds)}"


def format_status_line(snapshot: BatchDisplaySnapshot) -> str:
    """Single-line status for non-terminal environments."""
    return (
        f"[{format_duration(snapshot.elapsed_seconds)}] "
        f"{snapshot.status:<12} {snapshot.batch_id[-8:]} "
        f"processed {snapshot.processed_requests}/{snapshot.total_requests} "
        f"{snapshot.percent_complete:5.1f}% "
        f"ok {snapshot.completed_requests} failed {snapshot.failed_requests} "
        f"rem {snapshot.remaining_requests} "
        f"{format_speed(snapshot.speed_items_per_sec)} "
        f"{format_eta(snapshot.eta_seconds)}"
    )


def build_display_snapshot(
    *,
    batch_id: str,
    status: str,
    total_requests: int,
    completed_requests: int,
    failed_requests: int,
    started_monotonic: float,
    state_started_monotonic: float,
    now_monotonic: float,
) -> BatchDisplaySnapshot:
    """Compute a :class:`BatchDisplaySnapshot` from raw counters."""
    processed = completed_requests + failed_requests
    total = max(total_requests, processed)
    remaining = max(0, total - processed)
    percent = (processed / total * 100.0) if total > 0 else 0.0
    elapsed = max(0.0, now_monotonic - started_monotonic)
    state_elapsed = max(0.0, now_monotonic - state_started_monotonic)
    speed = processed / elapsed if elapsed > 0 and processed > 0 else None
    eta = remaining / speed if speed and speed > 0 and remaining > 0 else None

    return BatchDisplaySnapshot(
        batch_id=batch_id,
        status=status,
        total_requests=total,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        processed_requests=processed,
        remaining_requests=remaining,
        percent_complete=percent,
        elapsed_seconds=elapsed,
        state_elapsed_seconds=state_elapsed,
        speed_items_per_sec=speed,
        eta_seconds=eta,
    )


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def print_pipeline_summary(results: list[dict], console: Console | None = None) -> None:
    """Print a Rich table summarising pipeline stage results."""
    con = console or Console()
    table = Table(title="Pipeline Summary")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Details")

    for r in results:
        status = r.get("status", "?")
        style = {"completed": "green", "failed": "red", "skipped": "dim"}.get(status, "")
        dur = format_duration(r.get("duration_ms", 0) / 1000)
        detail = r.get("error") or r.get("detail") or ""
        table.add_row(r.get("name", "?"), f"[{style}]{status}[/{style}]", dur, str(detail)[:80])

    con.print(table)


def print_batch_summary(summary: dict, console: Console | None = None) -> None:
    """Print a Rich table summarising batch execution results."""
    con = console or Console()
    table = Table(title="Batch Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        table.add_row(str(key), str(value))

    con.print(table)


def print_stage_header(stage_name: str, stage_num: int, total: int, console: Console | None = None) -> None:
    """Print a Rich panel announcing a pipeline stage."""
    con = console or Console()
    con.print(Panel(f"[bold]Stage {stage_num}/{total}: {stage_name}[/bold]"))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _progress_fields(snapshot: BatchDisplaySnapshot) -> dict[str, int | str]:
    return {
        "total": max(1, snapshot.total_requests),
        "completed": snapshot.processed_requests,
        "status": snapshot.status,
        "batch_suffix": snapshot.batch_id[-8:] if snapshot.batch_id else "",
        "ok": snapshot.completed_requests,
        "failed": snapshot.failed_requests,
        "remaining": snapshot.remaining_requests,
        "speed": format_speed(snapshot.speed_items_per_sec),
        "eta": format_eta(snapshot.eta_seconds),
    }


# ---------------------------------------------------------------------------
# Stage progress bar (tqdm-style)
# ---------------------------------------------------------------------------


def make_stage_progress(description: str, total: int, console: Console | None = None) -> tuple[Progress, int]:
    """Create a Rich progress bar for a pipeline stage.

    Returns ``(progress, task_id)``.  The caller must call
    ``progress.start()`` before the loop and ``progress.stop()`` after.
    The progress bar shows: description, bar, N/M, elapsed, ETA, and
    iterations/sec (or sec/iteration) in tqdm style.
    """
    con = console or Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]}"),
        console=con,
        transient=True,
    )
    task_id = progress.add_task(description, total=total, speed="?it/s")
    return progress, task_id


def update_stage_speed(progress: Progress, task_id: int) -> None:
    """Recompute and update the speed field on a stage progress bar."""
    task = progress.tasks[task_id]
    elapsed = task.elapsed
    completed = task.completed
    if elapsed and elapsed > 0 and completed > 0:
        speed = completed / elapsed
        progress.update(task_id, speed=format_speed(speed))
    else:
        progress.update(task_id, speed="?it/s")
