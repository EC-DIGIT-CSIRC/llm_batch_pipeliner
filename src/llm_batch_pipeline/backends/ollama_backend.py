"""Ollama local batch execution backend.

Translates OpenAI ``/v1/responses`` JSONL into Ollama ``/api/chat`` requests
and submits them via the shared :class:`HttpServerBackend` machinery (sharding,
retry, warmup, telemetry).

This module focuses on the **Ollama-specific** bits:

- Translating ``/v1/responses`` requests to Ollama ``/api/chat`` payloads.
- Sanitising JSON Schema for llama.cpp's GBNF grammar subset.
- Wrapping Ollama responses back into OpenAI-compatible records.

All sharding, parallel execution, retry, warmup, OTLP/Loki telemetry, and
output writing live in :class:`HttpServerBackend`.
"""

from __future__ import annotations

import uuid
from typing import Any

from llm_batch_pipeline.backends.http_server import (
    ExecutionResult,
    HttpServerBackend,
    PreparedRequest,
    _summarise_row_results,
)
from llm_batch_pipeline.config import OLLAMA_CHAT_ENDPOINT, BatchConfig

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
# Backwards-compatible aliases
# ---------------------------------------------------------------------------
# Older tests and downstream code import these names directly. Keep them
# working by re-exporting the shared dataclasses under their historical names.
OllamaPreparedRequest = PreparedRequest
OllamaExecutionResult = ExecutionResult


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OllamaBackend(HttpServerBackend):
    """Execute batch requests locally against one or more Ollama servers."""

    backend_name = "ollama"
    logger_name = "llm_batch_pipeline.backends.ollama"
    default_base_url = "http://localhost:11434"
    batch_id_prefix = "ollama_batch_"

    def endpoint_path(self) -> str:
        return OLLAMA_CHAT_ENDPOINT

    def translate_request(self, index: int, request: dict[str, Any], config: BatchConfig) -> PreparedRequest:
        return _translate_request(index, request, config.model)

    def warmup_payload(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

    def build_success_record(self, custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
        return _build_success_record(custom_id, resp)


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _translate_request(index: int, request: dict[str, Any], model: str) -> PreparedRequest:
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

    return PreparedRequest(
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


__all__ = [
    "OllamaBackend",
    "OllamaPreparedRequest",
    "OllamaExecutionResult",
    "_translate_request",
    "_sanitise_schema_for_ollama",
    "_build_success_record",
    "_summarise_row_results",
]
