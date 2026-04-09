"""Shared batch submission infrastructure.

Provides validation, progress display helpers, atomic I/O, and summary
construction used by both the OpenAI and Ollama backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_batch_pipeline.config import OPENAI_BATCH_ENDPOINT

logger = logging.getLogger("llm_batch_pipeline.backends.common")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchValidationResult:
    """Summary of JSONL batch validation."""

    total_requests: int
    model: str
    custom_ids: list[str]


def load_and_validate_batch(jsonl_path: Path) -> tuple[list[dict[str, Any]], BatchValidationResult]:
    """Load a batch JSONL file, validate structure, and return (requests, result).

    Checks:
    - Each line parses as JSON.
    - Each request has ``custom_id``, ``method: POST``, ``url``, ``body.model``.
    - All ``custom_id`` values are unique.
    - All requests target the same model.
    """
    text = jsonl_path.read_text(encoding="utf-8")
    requests: list[dict[str, Any]] = []
    custom_ids: list[str] = []
    seen_ids: set[str] = set()
    model: str = ""

    for line_num, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"Line {line_num}: invalid JSON: {exc}"
            raise ValueError(msg) from exc

        cid = obj.get("custom_id")
        if not cid:
            msg = f"Line {line_num}: missing 'custom_id'"
            raise ValueError(msg)
        if cid in seen_ids:
            msg = f"Line {line_num}: duplicate custom_id '{cid}'"
            raise ValueError(msg)
        seen_ids.add(cid)

        if obj.get("method", "").upper() != "POST":
            msg = f"Line {line_num}: method must be POST"
            raise ValueError(msg)
        if obj.get("url") != OPENAI_BATCH_ENDPOINT:
            msg = f"Line {line_num}: url must be '{OPENAI_BATCH_ENDPOINT}'"
            raise ValueError(msg)

        body = obj.get("body", {})
        line_model = body.get("model", "")
        if not line_model:
            msg = f"Line {line_num}: missing body.model"
            raise ValueError(msg)
        if model and line_model != model:
            msg = f"Line {line_num}: mixed models ({model!r} vs {line_model!r})"
            raise ValueError(msg)
        model = line_model

        custom_ids.append(cid)
        requests.append(obj)

    if not requests:
        msg = "Batch file is empty"
        raise ValueError(msg)

    return requests, BatchValidationResult(
        total_requests=len(requests),
        model=model,
        custom_ids=custom_ids,
    )


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------


def write_text_atomic(path: Path, text: str) -> None:
    """Write atomically: temp file then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically."""
    write_text_atomic(path, json.dumps(data, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Status history
# ---------------------------------------------------------------------------


def append_status_history(path: Path, entry: dict[str, Any]) -> None:
    """Append a timestamped JSONL line to *path*."""
    entry.setdefault("timestamp", time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Prompt override
# ---------------------------------------------------------------------------


def resolve_prompt_override(
    prompt: str | None,
    prompt_file: Path | None,
) -> str | None:
    """Return prompt text from either direct string or file."""
    if prompt:
        return prompt
    if prompt_file and prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8").strip()
    return None


def apply_prompt_override(requests: list[dict[str, Any]], new_prompt: str) -> None:
    """Rewrite ``body.instructions`` in every request in-place."""
    for req in requests:
        req.setdefault("body", {})["instructions"] = new_prompt


# ---------------------------------------------------------------------------
# Summary construction
# ---------------------------------------------------------------------------


def build_summary(
    *,
    batch_id: str,
    status: str,
    total_requests: int,
    completed_requests: int,
    failed_requests: int,
    started_at: float,
    finished_at: float,
    model: str = "",
    source_file: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a standard summary dict."""
    summary: dict[str, Any] = {
        "batch_id": batch_id,
        "status": status,
        "model": model,
        "source_file": source_file,
        "total_requests": total_requests,
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "duration_seconds": round(finished_at - started_at, 3),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if total_requests > 0:
        duration = finished_at - started_at
        if duration > 0:
            summary["requests_per_second"] = round(completed_requests / duration, 3)
    if extra:
        summary.update(extra)
    return summary


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
