"""OpenAI Batch API backend.

Uploads a JSONL file, creates a batch, polls until terminal state,
and downloads output/error files.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from llm_batch_pipeline.backends.base import BatchBackend, SubmissionResult
from llm_batch_pipeline.backends.common import (
    append_status_history,
    apply_prompt_override,
    build_summary,
    load_and_validate_batch,
    resolve_prompt_override,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from llm_batch_pipeline.config import OPENAI_BATCH_ENDPOINT, BatchConfig
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.tui import (
    RichStatusReporter,
    StatusReporter,
    build_display_snapshot,
    print_batch_summary,
)

logger = logging.getLogger("llm_batch_pipeline.backends.openai")

_TERMINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


class OpenAIBackend(BatchBackend):
    """Submit batches via the OpenAI Batch API."""

    @property
    def name(self) -> str:
        return "openai"

    def submit(
        self,
        batch_jsonl: Path,
        config: BatchConfig,
        *,
        console: Console | None = None,
        reporter: StatusReporter | None = None,
        metrics: MetricsCollector | None = None,
        client: Any | None = None,
        sleep_fn: Any | None = None,
    ) -> SubmissionResult:
        con = console or Console()
        rep = reporter or RichStatusReporter(console=con)
        sleep = sleep_fn or time.sleep
        metrics = metrics or MetricsCollector()

        try:
            import openai as openai_mod  # noqa: PLC0415
        except ImportError as exc:
            msg = "openai package is required for the OpenAI backend"
            raise ImportError(msg) from exc

        oai_client = client or openai_mod.OpenAI()

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "status_history.jsonl"

        # Validate batch
        requests, validation = load_and_validate_batch(batch_jsonl)
        log_event(
            logger,
            f"Validated {validation.total_requests} requests, model={validation.model}",
            step="submit",
            status="validated",
            total=validation.total_requests,
        )

        # Apply prompt override if given
        prompt_text = resolve_prompt_override(config.prompt_override, config.prompt_override_file)
        if prompt_text:
            apply_prompt_override(requests, prompt_text)
            override_path = output_dir / "batch_submitted.jsonl"
            write_text_atomic(override_path, "\n".join(json.dumps(r, ensure_ascii=True) for r in requests) + "\n")
            batch_jsonl = override_path
            log_event(logger, "Applied prompt override", step="submit", status="prompt_override")

        # Resume or submit new
        if config.resume_batch_id:
            batch_id = config.resume_batch_id
            log_event(logger, f"Resuming batch {batch_id}", step="submit", status="resuming")
        else:
            # Upload file
            source_hash = sha256_file(batch_jsonl)
            with open(batch_jsonl, "rb") as fh:
                file_obj = oai_client.files.create(file=fh, purpose="batch")

            # Create batch
            batch_obj = oai_client.batches.create(
                input_file_id=file_obj.id,
                endpoint=OPENAI_BATCH_ENDPOINT,
                completion_window=config.completion_window,
            )
            batch_id = batch_obj.id

            # Save submission metadata
            write_json_atomic(
                output_dir / "submission.json",
                {
                    "batch_id": batch_id,
                    "file_id": file_obj.id,
                    "source_file": str(batch_jsonl),
                    "source_hash": source_hash,
                    "total_requests": validation.total_requests,
                    "model": validation.model,
                    "submitted_at": time.time(),
                },
            )
            log_event(
                logger,
                f"Submitted batch {batch_id}",
                step="submit",
                status="submitted",
                batch_id=batch_id,
                file_id=file_obj.id,
            )

        if config.no_wait:
            return SubmissionResult(
                batch_id=batch_id,
                status="submitted",
                total_requests=validation.total_requests,
            )

        # Poll until terminal
        started = time.monotonic()
        state_started = started

        result = self._poll_until_terminal(
            oai_client,
            batch_id,
            config,
            validation,
            rep=rep,
            sleep=sleep,
            started_monotonic=started,
            state_started_monotonic=state_started,
            history_path=history_path,
            metrics=metrics,
        )

        finished_at = time.time()
        duration = time.monotonic() - started

        # Download output files
        batch_obj = oai_client.batches.retrieve(batch_id)
        output_path = self._download_file(oai_client, batch_obj, "output_file_id", output_dir / "output.jsonl")
        error_path = self._download_file(oai_client, batch_obj, "error_file_id", output_dir / "errors.jsonl")

        summary = build_summary(
            batch_id=batch_id,
            status=result,
            total_requests=validation.total_requests,
            completed_requests=self._get_count(batch_obj, "completed"),
            failed_requests=self._get_count(batch_obj, "failed"),
            started_at=time.time() - duration,
            finished_at=finished_at,
            model=validation.model,
            source_file=str(batch_jsonl),
        )
        write_json_atomic(output_dir / "summary.json", summary)
        print_batch_summary(summary, console=con)

        return SubmissionResult(
            batch_id=batch_id,
            status=result,
            output_file=output_path,
            error_file=error_path,
            total_requests=validation.total_requests,
            completed_requests=self._get_count(batch_obj, "completed"),
            failed_requests=self._get_count(batch_obj, "failed"),
            duration_seconds=duration,
            summary=summary,
        )

    # -- polling -------------------------------------------------------------

    def _poll_until_terminal(
        self,
        client: Any,
        batch_id: str,
        config: BatchConfig,
        validation: Any,
        *,
        rep: StatusReporter,
        sleep: Any,
        started_monotonic: float,
        state_started_monotonic: float,
        history_path: Path,
        metrics: MetricsCollector,
    ) -> str:
        last_status = ""
        while True:
            batch_obj = client.batches.retrieve(batch_id)
            status = batch_obj.status
            now = time.monotonic()

            if status != last_status:
                state_started_monotonic = now
                append_status_history(history_path, {"status": status, "batch_id": batch_id})
                last_status = status

            completed = self._get_count(batch_obj, "completed")
            failed = self._get_count(batch_obj, "failed")

            snapshot = build_display_snapshot(
                batch_id=batch_id,
                status=status,
                total_requests=validation.total_requests,
                completed_requests=completed,
                failed_requests=failed,
                started_monotonic=started_monotonic,
                state_started_monotonic=state_started_monotonic,
                now_monotonic=now,
            )

            if last_status == status and snapshot.processed_requests == 0:
                rep.start(snapshot)
            else:
                rep.update(snapshot)

            if status in _TERMINAL_STATES:
                rep.stop(snapshot)
                return status

            sleep(config.poll_interval_seconds)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _get_count(batch_obj: Any, field: str) -> int:
        counts = getattr(batch_obj, "request_counts", None)
        if counts is None:
            return 0
        return getattr(counts, field, 0)

    @staticmethod
    def _download_file(client: Any, batch_obj: Any, attr: str, dest: Path) -> Path | None:
        file_id = getattr(batch_obj, attr, None)
        if not file_id:
            return None
        try:
            content = client.files.content(file_id)
            data = _content_to_bytes(content)
            write_text_atomic(dest, data.decode("utf-8"))
            return dest
        except Exception:  # noqa: BLE001
            logger.warning("Could not download %s", attr)
            return None


def _content_to_bytes(content: Any) -> bytes:
    """Extract raw bytes from an OpenAI file content response."""
    if isinstance(content, bytes):
        return content
    if hasattr(content, "text"):
        return content.text.encode("utf-8")
    if hasattr(content, "content"):
        return content.content if isinstance(content.content, bytes) else content.content.encode("utf-8")
    if hasattr(content, "read"):
        return content.read()
    return str(content).encode("utf-8")
