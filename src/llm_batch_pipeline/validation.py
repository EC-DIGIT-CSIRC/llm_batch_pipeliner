"""Pydantic schema validation of LLM result rows.

Reads the batch output JSONL, parses each row's response text as JSON,
and validates it against the ``mySchema`` Pydantic model.  Produces
per-row pass/fail results with error messages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.schema_loader import load_schema_class

logger = logging.getLogger("llm_batch_pipeline.validation")


@dataclass(slots=True)
class ValidationRow:
    """Result of validating a single output row."""

    custom_id: str
    valid: bool
    parsed_data: dict[str, Any] | None = None
    error_message: str | None = None
    raw_text: str = ""


@dataclass(slots=True)
class ValidationResult:
    """Aggregate validation results."""

    rows: list[ValidationRow] = field(default_factory=list)
    total: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    skipped_count: int = 0  # rows with errors in batch output

    @property
    def valid_rows(self) -> list[ValidationRow]:
        return [r for r in self.rows if r.valid]

    @property
    def invalid_rows(self) -> list[ValidationRow]:
        return [r for r in self.rows if not r.valid]


def validate_batch_output(
    output_jsonl: Path,
    schema_path: Path | None = None,
) -> ValidationResult:
    """Validate every row in *output_jsonl* against *schema_path*.

    If *schema_path* is ``None``, validation is skipped and all parseable
    rows are marked as valid.
    """
    schema_cls = load_schema_class(schema_path) if schema_path else None
    result = ValidationResult()

    text = output_jsonl.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        result.total += 1
        record = json.loads(stripped)
        custom_id = record.get("custom_id", "")

        # Skip error records
        if record.get("error") is not None:
            result.skipped_count += 1
            result.rows.append(
                ValidationRow(
                    custom_id=custom_id,
                    valid=False,
                    error_message=f"Batch error: {record['error']}",
                )
            )
            continue

        # Extract output text
        raw_text = _extract_output_text(record)
        if raw_text is None:
            result.skipped_count += 1
            result.rows.append(
                ValidationRow(
                    custom_id=custom_id,
                    valid=False,
                    error_message="Could not extract output text from response",
                )
            )
            continue

        # Parse JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            result.invalid_count += 1
            result.rows.append(
                ValidationRow(
                    custom_id=custom_id,
                    valid=False,
                    error_message=f"JSON parse error: {exc}",
                    raw_text=raw_text,
                )
            )
            continue

        # Validate against schema
        if schema_cls is not None:
            try:
                schema_cls.model_validate(parsed)
                result.valid_count += 1
                result.rows.append(
                    ValidationRow(
                        custom_id=custom_id,
                        valid=True,
                        parsed_data=parsed,
                        raw_text=raw_text,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                result.invalid_count += 1
                result.rows.append(
                    ValidationRow(
                        custom_id=custom_id,
                        valid=False,
                        error_message=f"Schema validation error: {exc}",
                        parsed_data=parsed,
                        raw_text=raw_text,
                    )
                )
        else:
            # No schema — accept any valid JSON
            result.valid_count += 1
            result.rows.append(
                ValidationRow(
                    custom_id=custom_id,
                    valid=True,
                    parsed_data=parsed,
                    raw_text=raw_text,
                )
            )

    log_event(
        logger,
        f"Validation complete: {result.valid_count} valid, "
        f"{result.invalid_count} invalid, {result.skipped_count} skipped",
        step="validate",
        status="ok",
        total=result.total,
        valid=result.valid_count,
        invalid=result.invalid_count,
        skipped=result.skipped_count,
    )
    return result


def _extract_output_text(record: dict[str, Any]) -> str | None:
    """Extract the output text from an OpenAI-compatible response record."""
    candidate: str | None = None
    try:
        body = record["response"]["body"]
        for output_item in body.get("output", []):
            if output_item.get("type") == "message":
                for content_item in output_item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text = content_item.get("text")
                        if text:
                            return text
                        if text == "":
                            candidate = text
    except (KeyError, TypeError):
        pass
    return candidate
