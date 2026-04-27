"""Unit tests for the vLLM batch backend.

Covers:
- /v1/responses request passthrough (default endpoint)
- /v1/chat/completions request translation
- Auth header presence iff API key set
- Warmup payload shape per endpoint
- Response parsing for both endpoints
- Configure() validation
"""

from __future__ import annotations

from llm_batch_pipeline.backends.vllm_backend import (
    VllmBackend,
    _responses_body_to_chat_completions,
    _wrap_chat_completion_response,
    _wrap_responses_response,
)
from llm_batch_pipeline.config import VLLM_CHAT_ENDPOINT, VLLM_RESPONSES_ENDPOINT, BatchConfig


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------


class TestEndpointSelection:
    def test_default_endpoint_is_responses(self):
        backend = VllmBackend()
        assert backend.endpoint_path() == VLLM_RESPONSES_ENDPOINT

    def test_configure_chat(self):
        backend = VllmBackend().configure(endpoint="chat")
        assert backend.endpoint_path() == VLLM_CHAT_ENDPOINT

    def test_configure_invalid_endpoint_raises(self):
        backend = VllmBackend()
        try:
            backend.configure(endpoint="bogus")
        except ValueError as exc:
            assert "bogus" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError")


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    def test_no_header_when_no_key(self):
        backend = VllmBackend()
        cfg = BatchConfig(api_key=None)
        assert backend.auth_headers(cfg) == {}

    def test_bearer_header_from_config(self):
        backend = VllmBackend()
        cfg = BatchConfig(api_key="abc123")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer abc123"}

    def test_bearer_header_from_env(self, monkeypatch):
        backend = VllmBackend()
        cfg = BatchConfig(api_key=None)
        monkeypatch.setenv("VLLM_API_KEY", "envtoken")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer envtoken"}

    def test_config_overrides_env(self, monkeypatch):
        backend = VllmBackend()
        cfg = BatchConfig(api_key="explicit")
        monkeypatch.setenv("VLLM_API_KEY", "envtoken")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer explicit"}


# ---------------------------------------------------------------------------
# Warmup payloads
# ---------------------------------------------------------------------------


class TestWarmupPayload:
    def test_responses_warmup(self):
        backend = VllmBackend()
        payload = backend.warmup_payload("gemma4:latest")
        assert payload["model"] == "gemma4:latest"
        assert payload["stream"] is False
        assert isinstance(payload["input"], list)
        assert payload["input"][0]["role"] == "user"

    def test_chat_warmup(self):
        backend = VllmBackend().configure(endpoint="chat")
        payload = backend.warmup_payload("gemma4:latest")
        assert payload["model"] == "gemma4:latest"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


SAMPLE_RENDERED_REQUEST = {
    "custom_id": "row-7",
    "method": "POST",
    "url": "/v1/responses",
    "body": {
        "model": "gemma4:latest",
        "instructions": "Classify the email as spam or ham.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello world"}]},
        ],
        "temperature": 0,
        "seed": 42,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "spam_result",
                "schema": {"type": "object", "properties": {"label": {"type": "string"}}},
                "strict": True,
            }
        },
    },
}


class TestTranslateRequest:
    def test_responses_passthrough(self):
        backend = VllmBackend()
        cfg = BatchConfig(model="gemma4:latest")
        prep = backend.translate_request(7, SAMPLE_RENDERED_REQUEST, cfg)
        assert prep.index == 7
        assert prep.custom_id == "row-7"
        # /v1/responses passthrough: payload is the rendered body unchanged
        assert prep.payload == SAMPLE_RENDERED_REQUEST["body"]

    def test_chat_translation(self):
        backend = VllmBackend().configure(endpoint="chat")
        cfg = BatchConfig(model="gemma4:latest")
        prep = backend.translate_request(7, SAMPLE_RENDERED_REQUEST, cfg)
        assert prep.index == 7
        assert prep.custom_id == "row-7"

        payload = prep.payload
        assert payload["model"] == "gemma4:latest"
        assert payload["stream"] is False
        # System + user message in chat shape
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Classify the email as spam or ham.",
        }
        assert payload["messages"][1] == {"role": "user", "content": "Hello world"}
        # Sampling fields preserved (reproducibility)
        assert payload["temperature"] == 0
        assert payload["seed"] == 42
        # Structured outputs converted to chat-completions response_format
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "spam_result"
        assert payload["response_format"]["json_schema"]["strict"] is True

    def test_chat_translation_uses_config_model_when_missing(self):
        body_no_model = dict(SAMPLE_RENDERED_REQUEST["body"])
        body_no_model.pop("model")
        req = dict(SAMPLE_RENDERED_REQUEST)
        req["body"] = body_no_model

        cfg = BatchConfig(model="fallback-model")
        payload = _responses_body_to_chat_completions(body_no_model, default_model=cfg.model)
        assert payload["model"] == "fallback-model"


# ---------------------------------------------------------------------------
# Response wrappers
# ---------------------------------------------------------------------------


class TestResponseWrappers:
    def test_responses_wrap_attaches_custom_id_and_envelope(self):
        resp = {
            "id": "resp_xyz",
            "model": "gemma4:latest",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"label": "spam"}'}],
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        wrapped = _wrap_responses_response("row-7", resp)
        assert wrapped["custom_id"] == "row-7"
        assert wrapped["error"] is None
        assert wrapped["response"]["status_code"] == 200
        assert wrapped["response"]["request_id"] == "resp_xyz"
        body = wrapped["response"]["body"]
        # Body keeps original output array
        assert body["output"][0]["content"][0]["text"] == '{"label": "spam"}'
        # provider_meta.vllm is added with usage info
        assert body["provider_meta"]["vllm"]["model"] == "gemma4:latest"
        assert body["provider_meta"]["vllm"]["total_tokens"] == 60

    def test_chat_wrap_translates_choices_to_responses_shape(self):
        resp = {
            "id": "chatcmpl_abc",
            "model": "gemma4:latest",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"label": "ham"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        }
        wrapped = _wrap_chat_completion_response("row-3", resp)
        assert wrapped["custom_id"] == "row-3"
        assert wrapped["error"] is None

        body = wrapped["response"]["body"]
        # Wrapped into /v1/responses-style output[]
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == '{"label": "ham"}'
        # Token usage normalised to input/output_tokens
        assert body["usage"]["input_tokens"] == 30
        assert body["usage"]["output_tokens"] == 5

    def test_chat_wrap_handles_empty_choices(self):
        wrapped = _wrap_chat_completion_response("row-x", {"id": "x"})
        body = wrapped["response"]["body"]
        assert body["output"][0]["content"][0]["text"] == ""


# ---------------------------------------------------------------------------
# Build success record dispatches on endpoint kind
# ---------------------------------------------------------------------------


class TestBuildSuccessRecord:
    def test_default_uses_responses_wrapper(self):
        backend = VllmBackend()
        resp = {"id": "r1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]}
        rec = backend.build_success_record("cid", resp)
        assert rec["response"]["body"]["output"][0]["content"][0]["text"] == "x"

    def test_chat_kind_uses_chat_wrapper(self):
        backend = VllmBackend().configure(endpoint="chat")
        resp = {
            "id": "c1",
            "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        rec = backend.build_success_record("cid", resp)
        assert rec["response"]["body"]["output"][0]["content"][0]["text"] == "answer"


# ---------------------------------------------------------------------------
# Backend metadata
# ---------------------------------------------------------------------------


class TestBackendMetadata:
    def test_name(self):
        assert VllmBackend().name == "vllm"

    def test_logger_name(self):
        assert VllmBackend().logger_name == "llm_batch_pipeline.backends.vllm"

    def test_batch_id_prefix(self):
        assert VllmBackend().batch_id_prefix == "vllm_batch_"
