"""Generic HTTP-server batch backend.

Encapsulates the parts of the Ollama backend that are not Ollama-specific:
multi-server sharding, per-shard thread pools, retry/backoff, warmup,
row-summary statistics, output writing, and OTLP/Loki telemetry parity.

Subclasses (``OllamaBackend``, ``VllmBackend``, ...) only implement the
small set of engine-specific hooks defined on this base class. This guarantees
that every HTTP-server-style backend emits **identical** structured log fields
and Prometheus metric labels, which keeps Grafana dashboards portable across
backends via a single ``backend`` label.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from llm_batch_pipeline.backends.base import BatchBackend, SubmissionResult
from llm_batch_pipeline.backends.common import (
    apply_prompt_override,
    build_summary,
    load_and_validate_batch,
    resolve_prompt_override,
    write_text_atomic,
)
from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.tui import (
    RichStatusReporter,
    StatusReporter,
    build_display_snapshot,
    print_batch_summary,
)

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0
_WARMUP_TIMEOUT_SECONDS = 300  # cold-start model loads can take 30-120+ seconds


@dataclass(slots=True)
class PreparedRequest:
    """A single prepared per-row request, indexed by original JSONL position."""

    index: int
    custom_id: str
    payload: dict[str, Any]


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of a single executed request."""

    index: int
    custom_id: str
    success_record: dict[str, Any] | None = None
    error_record: dict[str, Any] | None = None
    duration_ms: float = 0.0


class HttpServerBackend(BatchBackend):
    """Base class for backends that talk to a self-hosted HTTP LLM server.

    Subclasses must override :attr:`backend_name`, :attr:`logger_name`,
    :meth:`endpoint_path`, :meth:`translate_request`, :meth:`build_success_record`,
    and :meth:`warmup_payload`.
    """

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    #: Short backend identifier. Used for metric labels and the ``"backend"``
    #: field in ``summary.json``. Must be unique across backends.
    backend_name: str = ""

    #: Python logger name. Subclasses set this to e.g.
    #: ``"llm_batch_pipeline.backends.ollama"``.
    logger_name: str = "llm_batch_pipeline.backends.http"

    #: Default port shown in fallback URL when ``--base-url`` is not set.
    default_base_url: str = "http://localhost:8000"

    #: Prefix for the deterministic batch ID (e.g. ``"ollama_batch_"``).
    batch_id_prefix: str = "http_batch_"

    @property
    def name(self) -> str:
        return self.backend_name

    def endpoint_path(self) -> str:
        """Server-relative API path (e.g. ``"/api/chat"`` or ``"/v1/responses"``)."""
        raise NotImplementedError

    def translate_request(self, index: int, request: dict[str, Any], config: BatchConfig) -> PreparedRequest:
        """Convert one rendered JSONL request into the engine's payload shape."""
        raise NotImplementedError

    def build_success_record(self, custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
        """Wrap a successful HTTP response body into the OpenAI-compatible
        ``output.jsonl`` row shape used by the rest of the pipeline.
        """
        raise NotImplementedError

    def warmup_payload(self, model: str) -> dict[str, Any]:
        """Body for a one-shot warmup request that forces the model to load."""
        raise NotImplementedError

    def auth_headers(self, config: BatchConfig) -> dict[str, str]:
        """Optional auth headers (e.g. ``Authorization: Bearer <token>``).

        Default: no headers. Subclasses can override.
        """
        return {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def submit(
        self,
        batch_jsonl: Path,
        config: BatchConfig,
        *,
        console: Console | None = None,
        reporter: StatusReporter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> SubmissionResult:
        log = logging.getLogger(self.logger_name)
        con = console or Console()
        rep = reporter or RichStatusReporter(console=con)
        met = metrics or MetricsCollector()

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load + validate ----
        requests, validation = load_and_validate_batch(batch_jsonl)
        log_event(
            log,
            f"Validated {validation.total_requests} requests for {self.backend_name}",
            step="submit",
            status="validated",
            backend=self.backend_name,
            total=validation.total_requests,
        )

        # ---- Prompt override ----
        prompt_text = resolve_prompt_override(config.prompt_override, config.prompt_override_file)
        if prompt_text:
            apply_prompt_override(requests, prompt_text)

        # ---- Translate to engine-native format ----
        prepared = [self.translate_request(idx, req, config) for idx, req in enumerate(requests)]

        # ---- Determine sharding ----
        base_urls = config.base_urls or [self.default_base_url]
        num_shards = config.num_shards or len(base_urls)
        shards: list[list[PreparedRequest]] = [[] for _ in range(num_shards)]
        for i, prep in enumerate(prepared):
            shards[i % num_shards].append(prep)

        batch_id = self._make_batch_id(batch_jsonl)

        # ---- Warm up each unique server ----
        unique_urls = list(dict.fromkeys(base_urls))
        for warmup_url in unique_urls:
            self._warmup_server(warmup_url, config, log)

        # ---- Execute ----
        started = time.monotonic()
        started_at = time.time()
        completed_count = 0
        failed_count = 0
        all_results: list[ExecutionResult] = []

        snapshot = build_display_snapshot(
            batch_id=batch_id,
            status="starting",
            total_requests=validation.total_requests,
            completed_requests=0,
            failed_requests=0,
            started_monotonic=started,
            state_started_monotonic=started,
            now_monotonic=time.monotonic(),
        )
        rep.start(snapshot)

        total_workers = num_shards * config.num_parallel_jobs
        with ThreadPoolExecutor(max_workers=total_workers) as pool:
            futures: dict[Any, PreparedRequest] = {}
            max_shard_len = max((len(s) for s in shards), default=0)
            # Round-robin submission so all shards get work concurrently
            for item_idx in range(max_shard_len):
                for shard_idx, shard_requests in enumerate(shards):
                    if item_idx >= len(shard_requests):
                        continue
                    base_url = base_urls[shard_idx % len(base_urls)]
                    prep = shard_requests[item_idx]
                    fut = pool.submit(self._execute_request, prep, base_url, config, met, log)
                    futures[fut] = prep

            for fut in as_completed(futures):
                exec_result = fut.result()
                all_results.append(exec_result)
                if exec_result.success_record:
                    completed_count += 1
                else:
                    failed_count += 1

                now = time.monotonic()
                snapshot = build_display_snapshot(
                    batch_id=batch_id,
                    status="in_progress",
                    total_requests=validation.total_requests,
                    completed_requests=completed_count,
                    failed_requests=failed_count,
                    started_monotonic=started,
                    state_started_monotonic=started,
                    now_monotonic=now,
                )
                rep.update(snapshot)

        # ---- Final snapshot ----
        snapshot = build_display_snapshot(
            batch_id=batch_id,
            status="completed",
            total_requests=validation.total_requests,
            completed_requests=completed_count,
            failed_requests=failed_count,
            started_monotonic=started,
            state_started_monotonic=started,
            now_monotonic=time.monotonic(),
        )
        rep.stop(snapshot)

        # ---- Sort + summary log + output files ----
        all_results.sort(key=lambda r: r.index)

        row_summary = _summarise_row_results(all_results)
        log_event(
            log,
            f"Row summary: {row_summary['rows_success']}/{row_summary['rows_total']} succeeded",
            step="submit",
            status="row_summary",
            backend=self.backend_name,
            batch_id=batch_id,
            model=config.model,
            **row_summary,
        )

        output_lines: list[str] = []
        error_lines: list[str] = []
        for r in all_results:
            if r.success_record:
                output_lines.append(json.dumps(r.success_record, ensure_ascii=True))
            if r.error_record:
                error_lines.append(json.dumps(r.error_record, ensure_ascii=True))

        output_path = output_dir / "output.jsonl"
        write_text_atomic(output_path, "\n".join(output_lines) + "\n" if output_lines else "")

        error_path: Path | None = None
        if error_lines:
            error_path = output_dir / "errors.jsonl"
            write_text_atomic(error_path, "\n".join(error_lines) + "\n")

        finished_at = time.time()
        duration = time.monotonic() - started

        summary = build_summary(
            batch_id=batch_id,
            status="completed" if failed_count == 0 else "completed_with_errors",
            total_requests=validation.total_requests,
            completed_requests=completed_count,
            failed_requests=failed_count,
            started_at=started_at,
            finished_at=finished_at,
            model=config.model,
            source_file=str(batch_jsonl),
            extra={
                "backend": self.backend_name,
                "base_urls": base_urls,
                "num_shards": num_shards,
            },
        )
        from llm_batch_pipeline.backends.common import write_json_atomic  # noqa: PLC0415

        write_json_atomic(output_dir / "summary.json", summary)
        print_batch_summary(summary, console=con)

        return SubmissionResult(
            batch_id=batch_id,
            status=summary["status"],
            output_file=output_path,
            error_file=error_path,
            total_requests=validation.total_requests,
            completed_requests=completed_count,
            failed_requests=failed_count,
            duration_seconds=duration,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _warmup_server(self, base_url: str, config: BatchConfig, log: logging.Logger) -> bool:
        """Warm up a single server endpoint. Returns True on 2xx, False otherwise."""
        url = f"{base_url.rstrip('/')}{self.endpoint_path()}"
        payload = self.warmup_payload(config.model)
        timeout = httpx.Timeout(_WARMUP_TIMEOUT_SECONDS)
        verify = not config.insecure
        headers = self.auth_headers(config)

        log_event(
            log,
            f"Warming up model '{config.model}' on {base_url}",
            step="submit",
            status="warmup_start",
            backend=self.backend_name,
            base_url=base_url,
            model=config.model,
        )
        start_ns = time.perf_counter_ns()
        try:
            with httpx.Client(timeout=timeout, verify=verify) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            log_event(
                log,
                f"Warmup succeeded on {base_url} in {duration_ms:.0f}ms",
                step="submit",
                status="warmup_ok",
                backend=self.backend_name,
                base_url=base_url,
                model=config.model,
                duration_ms=duration_ms,
            )
            return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            log_event(
                log,
                f"Warmup failed on {base_url}: {exc}",
                step="submit",
                status="warmup_failed",
                backend=self.backend_name,
                base_url=base_url,
                model=config.model,
                error=str(exc),
                duration_ms=duration_ms,
                level=logging.WARNING,
            )
            return False

    def _execute_request(
        self,
        prep: PreparedRequest,
        base_url: str,
        config: BatchConfig,
        metrics: MetricsCollector,
        log: logging.Logger,
    ) -> ExecutionResult:
        url = f"{base_url.rstrip('/')}{self.endpoint_path()}"
        timeout = httpx.Timeout(config.request_timeout_seconds)
        verify = not config.insecure
        headers = self.auth_headers(config)

        with httpx.Client(timeout=timeout, verify=verify) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                start_ns = time.perf_counter_ns()
                metrics.inc_active(self.backend_name)
                try:
                    response = client.post(url, json=prep.payload, headers=headers)
                    response.raise_for_status()
                    resp_data = response.json()

                    duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    metrics.dec_active(self.backend_name)
                    metrics.record_request(self.backend_name, config.model, duration_ms, "success")

                    log_event(
                        log,
                        f"OK {prep.custom_id} in {duration_ms:.0f}ms",
                        step="submit",
                        status="ok",
                        backend=self.backend_name,
                        custom_id=prep.custom_id,
                        duration_ms=duration_ms,
                        url=url,
                        level=logging.DEBUG,
                    )
                    return ExecutionResult(
                        index=prep.index,
                        custom_id=prep.custom_id,
                        success_record=self.build_success_record(prep.custom_id, resp_data),
                        duration_ms=duration_ms,
                    )

                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    metrics.dec_active(self.backend_name)

                    if attempt < _MAX_RETRIES:
                        log_event(
                            log,
                            f"Retry {attempt}/{_MAX_RETRIES} for {prep.custom_id}: {exc}",
                            step="submit",
                            status="retrying",
                            backend=self.backend_name,
                            custom_id=prep.custom_id,
                            attempt=attempt,
                            url=url,
                        )
                        time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                    else:
                        metrics.record_request(self.backend_name, config.model, duration_ms, "failure")
                        log_event(
                            log,
                            f"Failed after {_MAX_RETRIES} attempts: {prep.custom_id}: {exc}",
                            step="submit",
                            status="failed",
                            backend=self.backend_name,
                            custom_id=prep.custom_id,
                            error=str(exc),
                            url=url,
                            level=logging.ERROR,
                        )
                        return ExecutionResult(
                            index=prep.index,
                            custom_id=prep.custom_id,
                            error_record=_build_error_record(prep.custom_id, exc),
                            duration_ms=duration_ms,
                        )
        # Unreachable; satisfies type checker
        return ExecutionResult(index=prep.index, custom_id=prep.custom_id)

    def _make_batch_id(self, path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        except OSError:
            h.update(str(path).encode())
        return f"{self.batch_id_prefix}{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_error_record(custom_id: str, exc: Exception) -> dict[str, Any]:
    """Build an error record in OpenAI-compatible format."""
    return {
        "custom_id": custom_id,
        "error": {
            "code": exc.__class__.__name__,
            "message": str(exc),
        },
        "response": None,
    }


def _summarise_row_results(results: list[ExecutionResult]) -> dict[str, int | float]:
    """Compute row-level timing and outcome stats for a completed batch."""
    durations = [result.duration_ms for result in results]
    rows_total = len(results)
    rows_success = sum(1 for result in results if result.success_record)
    rows_failed = rows_total - rows_success

    if not durations:
        return {
            "rows_total": rows_total,
            "rows_success": rows_success,
            "rows_failed": rows_failed,
            "row_duration_avg_ms": 0.0,
            "row_duration_p50_ms": 0.0,
            "row_duration_min_ms": 0.0,
            "row_duration_max_ms": 0.0,
        }

    return {
        "rows_total": rows_total,
        "rows_success": rows_success,
        "rows_failed": rows_failed,
        "row_duration_avg_ms": round(statistics.fmean(durations), 3),
        "row_duration_p50_ms": round(statistics.median(durations), 3),
        "row_duration_min_ms": round(min(durations), 3),
        "row_duration_max_ms": round(max(durations), 3),
    }
