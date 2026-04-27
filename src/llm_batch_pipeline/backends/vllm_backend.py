"""vLLM batch execution backend.

vLLM exposes an OpenAI-compatible HTTP server (``vllm serve``). We submit
JSONL requests to either:

* ``/v1/responses`` (default) — exact match for our rendered JSONL, no
  translation needed.
* ``/v1/chat/completions`` (opt-in via ``--vllm-endpoint chat``) — the
  more battle-tested API; we translate ``/v1/responses``-style rendered
  requests into chat-completion format.

Sharding across multiple ``vllm serve`` instances uses the same
``--base-url`` repeatable flag as the Ollama backend. See
``docs/running-vllm.md``.
"""

from __future__ import annotations

import os
from typing import Any

from llm_batch_pipeline.backends.http_server import HttpServerBackend, PreparedRequest
from llm_batch_pipeline.config import VLLM_CHAT_ENDPOINT, VLLM_RESPONSES_ENDPOINT, BatchConfig


class VllmBackend(HttpServerBackend):
    """Execute batch requests against one or more ``vllm serve`` endpoints."""

    backend_name = "vllm"
    logger_name = "llm_batch_pipeline.backends.vllm"
    default_base_url = "http://localhost:8000"
    batch_id_prefix = "vllm_batch_"

    def endpoint_path(self) -> str:
        # Default to /v1/responses (no translation needed); /v1/chat/completions
        # is the opt-in alternative.
        return VLLM_CHAT_ENDPOINT if self._endpoint_kind() == "chat" else VLLM_RESPONSES_ENDPOINT

    def auth_headers(self, config: BatchConfig) -> dict[str, str]:
        token = config.api_key or os.environ.get("VLLM_API_KEY") or ""
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def warmup_payload(self, model: str) -> dict[str, Any]:
        if self._endpoint_kind() == "chat":
            return {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        # /v1/responses
        return {
            "model": model,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
            "stream": False,
        }

    def translate_request(self, index: int, request: dict[str, Any], config: BatchConfig) -> PreparedRequest:
        custom_id = request.get("custom_id", f"request_{index}")
        body = dict(request.get("body", {}))

        # /v1/responses: vLLM accepts our rendered body essentially as-is.
        if self._endpoint_kind() != "chat":
            return PreparedRequest(index=index, custom_id=custom_id, payload=body)

        # /v1/chat/completions: translate from /v1/responses shape.
        return PreparedRequest(
            index=index,
            custom_id=custom_id,
            payload=_responses_body_to_chat_completions(body, default_model=config.model),
        )

    def build_success_record(self, custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
        if self._endpoint_kind() == "chat":
            return _wrap_chat_completion_response(custom_id, resp)
        return _wrap_responses_response(custom_id, resp)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _endpoint_kind(self) -> str:
        # Stash on instance so tests can read; fall back to config-style attr.
        return getattr(self, "_endpoint_kind_value", "responses")

    def configure(self, *, endpoint: str = "responses") -> "VllmBackend":
        """Select which API path to talk to. Returns ``self`` for chaining."""
        if endpoint not in {"responses", "chat"}:
            msg = f"Invalid vllm endpoint kind: {endpoint!r}; expected 'responses' or 'chat'"
            raise ValueError(msg)
        self._endpoint_kind_value = endpoint
        return self


# ---------------------------------------------------------------------------
# /v1/responses -> /v1/chat/completions translation
# ---------------------------------------------------------------------------


def _responses_body_to_chat_completions(body: dict[str, Any], *, default_model: str) -> dict[str, Any]:
    """Convert an OpenAI ``/v1/responses`` body to ``/v1/chat/completions`` shape."""
    messages: list[dict[str, Any]] = []

    instructions = body.get("instructions", "")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for input_item in body.get("input", []):
        role = input_item.get("role", "user")
        content_parts = input_item.get("content", [])
        text_parts: list[str] = []
        for part in content_parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "input_text":
                text_parts.append(part.get("text", ""))
        if text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    payload: dict[str, Any] = {
        "model": body.get("model", default_model),
        "messages": messages,
        "stream": False,
    }

    # Mirror common sampling fields if present (preserves reproducibility flags
    # like seed/temperature when they are set on the rendered request).
    for fwd in ("temperature", "top_p", "top_k", "seed", "max_tokens", "n"):
        if fwd in body:
            payload[fwd] = body[fwd]

    # Structured output: vLLM accepts response_format=json_schema natively.
    schema_format = body.get("text", {}).get("format", {})
    if schema_format.get("type") == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_format.get("name", "response"),
                "schema": schema_format.get("schema", {}),
                "strict": schema_format.get("strict", True),
            },
        }

    return payload


# ---------------------------------------------------------------------------
# Response wrappers (everything downstream expects /v1/responses shape)
# ---------------------------------------------------------------------------


def _wrap_responses_response(custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
    """vLLM /v1/responses already matches our downstream shape; just attach
    ``custom_id`` and the standard envelope.
    """
    body = dict(resp)
    body.setdefault("provider_meta", {})["vllm"] = _extract_vllm_usage(resp)
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": body,
            "request_id": resp.get("id", "vllm_unknown"),
            "status_code": 200,
        },
    }


def _wrap_chat_completion_response(custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
    """Translate a vLLM ``/v1/chat/completions`` response into the
    ``/v1/responses``-shaped envelope the rest of the pipeline expects.
    """
    choices = resp.get("choices", [])
    text = ""
    if choices:
        message = choices[0].get("message", {}) or {}
        text = message.get("content", "") or ""

    usage = resp.get("usage", {}) or {}
    body: dict[str, Any] = {
        "id": resp.get("id", "vllm_chat_unknown"),
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "provider_meta": {"vllm": _extract_vllm_usage(resp)},
    }
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": body,
            "request_id": resp.get("id", "vllm_chat_unknown"),
            "status_code": 200,
        },
    }


def _extract_vllm_usage(resp: dict[str, Any]) -> dict[str, Any]:
    """Pull commonly useful fields out of a vLLM response for ``provider_meta``."""
    usage = resp.get("usage", {}) or {}
    return {
        "model": resp.get("model", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


__all__ = [
    "VllmBackend",
    "_responses_body_to_chat_completions",
    "_wrap_chat_completion_response",
    "_wrap_responses_response",
]
