"""Batch JSONL renderer.

Reads input files (already parsed, filtered, and transformed), wraps them
in OpenAI Batch API request format, and shards into JSONL files under the
batch job's ``job/`` directory.

Each JSONL line is one request::

    {
        "custom_id": "<filename>",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "<model>",
            "instructions": "<prompt text>",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "..."}]}],
            "text": { ... }  // optional structured output schema
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from llm_batch_pipeline.config import OPENAI_BATCH_ENDPOINT, BatchConfig
from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.plugins.base import FileReader, ParsedFile
from llm_batch_pipeline.schema_loader import load_schema_format

logger = logging.getLogger("llm_batch_pipeline.render")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_batch(
    files: list[ParsedFile],
    reader: FileReader,
    config: BatchConfig,
) -> list[Path]:
    """Render *files* into sharded JSONL batch files.

    Returns the list of written shard file paths.
    """
    instructions = _read_instructions(config)
    schema_format = _load_optional_schema(config)

    lines: list[str] = []
    for pf in files:
        line = _render_line(pf, reader, config.model, instructions, schema_format)
        lines.append(line)

    shard_paths = _write_shards(
        lines,
        config.job_dir,
        max_requests=config.max_requests_per_shard,
        max_bytes=config.max_bytes_per_shard,
    )

    log_event(
        logger,
        f"Rendered {len(lines)} requests into {len(shard_paths)} shard(s)",
        step="render",
        status="ok",
        total_requests=len(lines),
        shards=len(shard_paths),
    )
    return shard_paths


# ---------------------------------------------------------------------------
# Line rendering
# ---------------------------------------------------------------------------


def _render_line(
    pf: ParsedFile,
    reader: FileReader,
    model: str,
    instructions: str,
    schema_format: dict[str, Any] | None,
) -> str:
    """Build one JSONL request line for a single file."""
    content_text = reader.package_for_llm(pf)

    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": content_text}],
            }
        ],
    }

    if schema_format is not None:
        body["text"] = schema_format

    request = {
        "custom_id": pf.filename,
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": body,
    }

    return json.dumps(request, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _write_shards(
    lines: list[str],
    job_dir: Path,
    *,
    max_requests: int,
    max_bytes: int,
) -> list[Path]:
    """Split *lines* into shards respecting request and byte limits."""
    job_dir.mkdir(parents=True, exist_ok=True)

    # Clean old shards
    for old in job_dir.glob("batch-*.jsonl"):
        old.unlink()
    # Remove convenience symlink
    symlink = job_dir / "batch.jsonl"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()

    shard_paths: list[Path] = []
    shard_lines: list[str] = []
    shard_bytes = 0
    shard_num = 0

    def _flush() -> None:
        nonlocal shard_num, shard_lines, shard_bytes
        if not shard_lines:
            return
        shard_num += 1
        path = job_dir / f"batch-{shard_num:05d}.jsonl"
        _write_text_atomic(path, "\n".join(shard_lines) + "\n")
        shard_paths.append(path)
        shard_lines = []
        shard_bytes = 0

    for line in lines:
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
        if shard_lines and (len(shard_lines) >= max_requests or shard_bytes + line_bytes > max_bytes):
            _flush()
        shard_lines.append(line)
        shard_bytes += line_bytes

    _flush()

    # Convenience symlink when exactly one shard
    if len(shard_paths) == 1:
        symlink.symlink_to(shard_paths[0].name)

    return shard_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_instructions(config: BatchConfig) -> str:
    """Load prompt text from config.  Falls back to a minimal default."""
    if config.prompt_file and config.prompt_file.is_file():
        return config.prompt_file.read_text(encoding="utf-8").strip()

    # Check batch_dir/prompt.txt
    candidate = config.batch_dir / "prompt.txt"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()

    return "Analyse the following content and respond according to the schema."


def _load_optional_schema(config: BatchConfig) -> dict[str, Any] | None:
    """Load schema if configured."""
    if config.schema_file and config.schema_file.is_file():
        return load_schema_format(config.schema_file)

    candidate = config.batch_dir / "schema.py"
    if candidate.is_file():
        return load_schema_format(candidate)

    return None


def _write_text_atomic(path: Path, text: str) -> None:
    """Write *text* atomically: write to ``.tmp``, then ``os.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
