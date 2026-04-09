"""Ollama local batch execution backend.

Translates OpenAI ``/v1/responses`` JSONL into Ollama ``/api/chat`` requests,
executes them locally via ``httpx`` with sharded multi-server parallelism,
and produces OpenAI-compatible output JSONL.

Key features:
- Multi-server sharding: round-robin distribution across ``--base-url`` endpoints.
- Per-shard thread pools for concurrent requests.
- JSON Schema sanitisation for llama.cpp GBNF grammar subset.
- Retry with linear backoff on connection/timeout errors.
- Output normalised to OpenAI response format for downstream compatibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
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
    write_json_atomic,
    write_text_atomic,
)
from llm_batch_pipeline.config import OLLAMA_CHAT_ENDPOINT, BatchConfig
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.tui import (
    RichStatusReporter,
    StatusReporter,
    build_display_snapshot,
    print_batch_summary,
)

logger = logging.getLogger("llm_batch_pipeline.backends.ollama")

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0
_WARMUP_TIMEOUT_SECONDS = 300  # 5 min for model loading on cold start

# JSON Schema keywords unsupported by llama.cpp GBNF grammar
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "maxLength",
        "minLength",
        "pattern",
        "format",
        "uniqueItems",
        "multipleOf",
    }
)


# ---------------------------------------------------------------------------
# Internal request types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OllamaPreparedRequest:
    index: int
    custom_id: str
    payload: dict[str, Any]


@dataclass(slots=True)
class OllamaExecutionResult:
    index: int
    custom_id: str
    success_record: dict[str, Any] | None = None
    error_record: dict[str, Any] | None = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OllamaBackend(BatchBackend):
    """Execute batch requests locally against Ollama server(s)."""

    @property
    def name(self) -> str:
        return "ollama"

    def submit(
        self,
        batch_jsonl: Path,
        config: BatchConfig,
        *,
        console: Console | None = None,
        reporter: StatusReporter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> SubmissionResult:
        con = console or Console()
        rep = reporter or RichStatusReporter(console=con)
        met = metrics or MetricsCollector()

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate
        requests, validation = load_and_validate_batch(batch_jsonl)
        log_event(
            logger,
            f"Validated {validation.total_requests} requests for Ollama",
            step="submit",
            status="validated",
            total=validation.total_requests,
        )

        # Apply prompt override
        prompt_text = resolve_prompt_override(config.prompt_override, config.prompt_override_file)
        if prompt_text:
            apply_prompt_override(requests, prompt_text)

        # Translate to Ollama format
        prepared = [_translate_request(idx, req, config.model) for idx, req in enumerate(requests)]

        # Determine sharding
        base_urls = config.base_urls or ["http://localhost:11434"]
        num_shards = config.num_shards or len(base_urls)
        shards: list[list[OllamaPreparedRequest]] = [[] for _ in range(num_shards)]
        for i, prep in enumerate(prepared):
            shards[i % num_shards].append(prep)

        # Generate batch ID
        batch_id = _make_batch_id(batch_jsonl)

        # Warm up each unique server to force model loading before the batch
        unique_urls = list(dict.fromkeys(base_urls))  # preserve order, dedupe
        for warmup_url in unique_urls:
            _warmup_server(warmup_url, config.model, insecure=config.insecure)

        # Execute shards
        started = time.monotonic()
        started_at = time.time()
        completed_count = 0
        failed_count = 0

        all_results: list[OllamaExecutionResult] = []

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
            futures = {}
            # Interleave submissions across shards so the thread pool
            # picks up work from all servers concurrently (round-robin).
            max_shard_len = max((len(s) for s in shards), default=0)
            for item_idx in range(max_shard_len):
                for shard_idx, shard_requests in enumerate(shards):
                    if item_idx >= len(shard_requests):
                        continue
                    base_url = base_urls[shard_idx % len(base_urls)]
                    prep = shard_requests[item_idx]
                    fut = pool.submit(
                        _execute_request,
                        prep,
                        base_url,
                        config,
                        met,
                    )
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

        # Final snapshot
        now = time.monotonic()
        snapshot = build_display_snapshot(
            batch_id=batch_id,
            status="completed",
            total_requests=validation.total_requests,
            completed_requests=completed_count,
            failed_requests=failed_count,
            started_monotonic=started,
            state_started_monotonic=started,
            now_monotonic=now,
        )
        rep.stop(snapshot)

        # Sort results by original index for deterministic output
        all_results.sort(key=lambda r: r.index)

        # Write output
        output_lines = []
        error_lines = []
        for r in all_results:
            if r.success_record:
                output_lines.append(json.dumps(r.success_record, ensure_ascii=True))
            if r.error_record:
                error_lines.append(json.dumps(r.error_record, ensure_ascii=True))

        output_path = output_dir / "output.jsonl"
        write_text_atomic(output_path, "\n".join(output_lines) + "\n" if output_lines else "")

        error_path = None
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
            extra={"backend": "ollama", "base_urls": base_urls, "num_shards": num_shards},
        )
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


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _translate_request(index: int, request: dict[str, Any], model: str) -> OllamaPreparedRequest:
    """Convert an OpenAI ``/v1/responses`` request to Ollama ``/api/chat`` format."""
    body = request.get("body", {})
    custom_id = request.get("custom_id", f"request_{index}")

    messages: list[dict[str, Any]] = []

    # System message from instructions
    instructions = body.get("instructions", "")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # User messages
    for input_item in body.get("input", []):
        role = input_item.get("role", "user")
        content_parts = input_item.get("content", [])
        text_parts = []
        for part in content_parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "input_text":
                text_parts.append(part.get("text", ""))
        if text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    ollama_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    # Schema for structured output
    schema_format = body.get("text", {}).get("format", {})
    if schema_format.get("type") == "json_schema":
        raw_schema = schema_format.get("schema", {})
        sanitised = _sanitise_schema_for_ollama(raw_schema)
        ollama_payload["format"] = sanitised

    return OllamaPreparedRequest(
        index=index,
        custom_id=custom_id,
        payload=ollama_payload,
    )


# ---------------------------------------------------------------------------
# Schema sanitisation
# ---------------------------------------------------------------------------


def _sanitise_schema_for_ollama(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema keywords unsupported by Ollama's GBNF grammar."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _sanitise_schema_for_ollama(v) for k, v in value.items()}
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            cleaned[key] = _sanitise_schema_for_ollama(value)
        elif key == "$defs" and isinstance(value, dict):
            cleaned[key] = {k: _sanitise_schema_for_ollama(v) for k, v in value.items()}
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            cleaned[key] = [_sanitise_schema_for_ollama(v) if isinstance(v, dict) else v for v in value]
        else:
            cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------


def _warmup_server(base_url: str, model: str, *, insecure: bool = False) -> bool:
    """Send a trivial chat request to force model loading before the batch.

    Returns ``True`` if the warmup succeeded, ``False`` otherwise.
    The warmup uses a generous timeout since the server may need to load
    the model into GPU VRAM (30-120+ seconds on cold start).
    """
    url = f"{base_url.rstrip('/')}{OLLAMA_CHAT_ENDPOINT}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    timeout = httpx.Timeout(_WARMUP_TIMEOUT_SECONDS)
    verify = not insecure

    log_event(
        logger,
        f"Warming up model '{model}' on {base_url}",
        step="submit",
        status="warmup_start",
        base_url=base_url,
        model=model,
    )
    start_ns = time.perf_counter_ns()

    try:
        with httpx.Client(timeout=timeout, verify=verify) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        log_event(
            logger,
            f"Warmup succeeded on {base_url} in {duration_ms:.0f}ms",
            step="submit",
            status="warmup_ok",
            base_url=base_url,
            model=model,
            duration_ms=duration_ms,
        )
        return True
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        log_event(
            logger,
            f"Warmup failed on {base_url}: {exc}",
            step="submit",
            status="warmup_failed",
            base_url=base_url,
            model=model,
            error=str(exc),
            duration_ms=duration_ms,
            level=logging.WARNING,
        )
        return False


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


def _execute_request(
    prep: OllamaPreparedRequest,
    base_url: str,
    config: BatchConfig,
    metrics: MetricsCollector,
) -> OllamaExecutionResult:
    """Execute a single Ollama request with retry logic."""
    url = f"{base_url.rstrip('/')}{OLLAMA_CHAT_ENDPOINT}"
    timeout = httpx.Timeout(config.request_timeout_seconds)
    verify = not config.insecure

    with httpx.Client(timeout=timeout, verify=verify) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            start_ns = time.perf_counter_ns()
            try:
                metrics.inc_active("ollama")

                response = client.post(url, json=prep.payload)
                response.raise_for_status()
                resp_data = response.json()

                duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                metrics.dec_active("ollama")
                metrics.record_request("ollama", config.model, duration_ms, "success")

                log_event(
                    logger,
                    f"OK {prep.custom_id} in {duration_ms:.0f}ms",
                    step="submit",
                    status="ok",
                    custom_id=prep.custom_id,
                    duration_ms=duration_ms,
                    url=url,
                    level=logging.DEBUG,
                )

                return OllamaExecutionResult(
                    index=prep.index,
                    custom_id=prep.custom_id,
                    success_record=_build_success_record(prep.custom_id, resp_data),
                    duration_ms=duration_ms,
                )

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                metrics.dec_active("ollama")

                if attempt < _MAX_RETRIES:
                    log_event(
                        logger,
                        f"Retry {attempt}/{_MAX_RETRIES} for {prep.custom_id}: {exc}",
                        step="submit",
                        status="retrying",
                        custom_id=prep.custom_id,
                        attempt=attempt,
                        url=url,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                else:
                    metrics.record_request("ollama", config.model, duration_ms, "failure")
                    log_event(
                        logger,
                        f"Failed after {_MAX_RETRIES} attempts: {prep.custom_id}: {exc}",
                        step="submit",
                        status="failed",
                        custom_id=prep.custom_id,
                        error=str(exc),
                        url=url,
                        level=logging.ERROR,
                    )
                    return OllamaExecutionResult(
                        index=prep.index,
                        custom_id=prep.custom_id,
                        error_record=_build_error_record(prep.custom_id, exc),
                        duration_ms=duration_ms,
                    )

    # Should not reach here, but satisfy type checker
    return OllamaExecutionResult(index=prep.index, custom_id=prep.custom_id)


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------


def _build_success_record(custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
    """Wrap an Ollama response in OpenAI-compatible format."""
    content = resp.get("message", {}).get("content", "")

    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": {
                "id": f"ollama_{uuid.uuid4().hex[:12]}",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ],
                "usage": {
                    "input_tokens": resp.get("prompt_eval_count", 0),
                    "output_tokens": resp.get("eval_count", 0),
                },
                "provider_meta": {
                    "ollama": {
                        "model": resp.get("model", ""),
                        "total_duration": resp.get("total_duration", 0),
                        "load_duration": resp.get("load_duration", 0),
                        "prompt_eval_duration": resp.get("prompt_eval_duration", 0),
                        "eval_duration": resp.get("eval_duration", 0),
                    }
                },
            },
            "request_id": f"ollama_{uuid.uuid4().hex[:12]}",
            "status_code": 200,
        },
    }


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


def _make_batch_id(path: Path) -> str:
    """Generate a deterministic batch ID from file content hash."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
    except OSError:
        h.update(str(path).encode())
    return f"ollama_batch_{h.hexdigest()[:16]}"
