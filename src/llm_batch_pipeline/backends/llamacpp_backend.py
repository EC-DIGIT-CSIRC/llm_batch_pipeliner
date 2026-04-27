"""llama.cpp batch execution backend.

llama.cpp exposes an OpenAI-compatible HTTP server (``llama-server``). We
submit JSONL requests to either:

* ``/v1/responses`` (default) - exact match for our rendered JSONL, no
  translation needed.
* ``/v1/chat/completions`` (opt-in via ``--llamacpp-endpoint chat``) - the
  more battle-tested API; we translate ``/v1/responses``-style rendered
  requests into chat-completion format.

This backend mirrors the vLLM backend, but keeps llama.cpp-specific logging
and provider metadata.
"""

from __future__ import annotations

import os
from typing import Any

from llm_batch_pipeline.backends.http_server import HttpServerBackend, PreparedRequest
from llm_batch_pipeline.config import LLAMACPP_CHAT_ENDPOINT, LLAMACPP_RESPONSES_ENDPOINT, BatchConfig


class LlamaCppBackend(HttpServerBackend):
    """Execute batch requests against one or more ``llama-server`` endpoints."""

    backend_name = "llamacpp"
    logger_name = "llm_batch_pipeline.backends.llamacpp"
    default_base_url = "http://localhost:8080"
    batch_id_prefix = "llamacpp_batch_"

    def endpoint_path(self) -> str:
        return LLAMACPP_CHAT_ENDPOINT if self._endpoint_kind() == "chat" else LLAMACPP_RESPONSES_ENDPOINT

    def auth_headers(self, config: BatchConfig) -> dict[str, str]:
        token = config.api_key or os.environ.get("LLAMA_API_KEY") or ""
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

        if self._endpoint_kind() != "chat":
            return PreparedRequest(index=index, custom_id=custom_id, payload=body)

        return PreparedRequest(
            index=index,
            custom_id=custom_id,
            payload=_responses_body_to_chat_completions(body, default_model=config.model),
        )

    def build_success_record(self, custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
        if self._endpoint_kind() == "chat":
            return _wrap_chat_completion_response(custom_id, resp)
        return _wrap_responses_response(custom_id, resp)

    def _endpoint_kind(self) -> str:
        return getattr(self, "_endpoint_kind_value", "responses")

    def configure(self, *, endpoint: str = "responses") -> "LlamaCppBackend":
        """Select which API path to talk to. Returns ``self`` for chaining."""
        if endpoint not in {"responses", "chat"}:
            msg = f"Invalid llama.cpp endpoint kind: {endpoint!r}; expected 'responses' or 'chat'"
            raise ValueError(msg)
        self._endpoint_kind_value = endpoint
        return self


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

    for fwd in ("temperature", "top_p", "top_k", "seed", "max_tokens", "n"):
        if fwd in body:
            payload[fwd] = body[fwd]

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


def _wrap_responses_response(custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
    body = dict(resp)
    body.setdefault("provider_meta", {})["llamacpp"] = _extract_llamacpp_usage(resp)
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": body,
            "request_id": resp.get("id", "llamacpp_unknown"),
            "status_code": 200,
        },
    }


def _wrap_chat_completion_response(custom_id: str, resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("choices", [])
    text = ""
    if choices:
        message = choices[0].get("message", {}) or {}
        text = message.get("content", "") or ""

    usage = resp.get("usage", {}) or {}
    body: dict[str, Any] = {
        "id": resp.get("id", "llamacpp_chat_unknown"),
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
        "provider_meta": {"llamacpp": _extract_llamacpp_usage(resp)},
    }
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": body,
            "request_id": resp.get("id", "llamacpp_chat_unknown"),
            "status_code": 200,
        },
    }


def _extract_llamacpp_usage(resp: dict[str, Any]) -> dict[str, Any]:
    usage = resp.get("usage", {}) or {}
    return {
        "model": resp.get("model", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


__all__ = [
    "LlamaCppBackend",
    "_responses_body_to_chat_completions",
    "_wrap_chat_completion_response",
    "_wrap_responses_response",
]
