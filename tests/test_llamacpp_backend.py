"""Unit tests for the llama.cpp batch backend."""

from __future__ import annotations

from llm_batch_pipeline.backends.llamacpp_backend import (
    LlamaCppBackend,
    _responses_body_to_chat_completions,
    _wrap_chat_completion_response,
    _wrap_responses_response,
)
from llm_batch_pipeline.config import LLAMACPP_CHAT_ENDPOINT, LLAMACPP_RESPONSES_ENDPOINT, BatchConfig


class TestEndpointSelection:
    def test_default_endpoint_is_responses(self):
        backend = LlamaCppBackend()
        assert backend.endpoint_path() == LLAMACPP_RESPONSES_ENDPOINT

    def test_configure_chat(self):
        backend = LlamaCppBackend().configure(endpoint="chat")
        assert backend.endpoint_path() == LLAMACPP_CHAT_ENDPOINT

    def test_configure_invalid_endpoint_raises(self):
        backend = LlamaCppBackend()
        try:
            backend.configure(endpoint="bogus")
        except ValueError as exc:
            assert "bogus" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError")


class TestAuthHeaders:
    def test_no_header_when_no_key(self):
        backend = LlamaCppBackend()
        cfg = BatchConfig(api_key=None)
        assert backend.auth_headers(cfg) == {}

    def test_bearer_header_from_config(self):
        backend = LlamaCppBackend()
        cfg = BatchConfig(api_key="abc123")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer abc123"}

    def test_bearer_header_from_env(self, monkeypatch):
        backend = LlamaCppBackend()
        cfg = BatchConfig(api_key=None)
        monkeypatch.setenv("LLAMA_API_KEY", "envtoken")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer envtoken"}

    def test_config_overrides_env(self, monkeypatch):
        backend = LlamaCppBackend()
        cfg = BatchConfig(api_key="explicit")
        monkeypatch.setenv("LLAMA_API_KEY", "envtoken")
        assert backend.auth_headers(cfg) == {"Authorization": "Bearer explicit"}


class TestWarmupPayload:
    def test_responses_warmup(self):
        backend = LlamaCppBackend()
        payload = backend.warmup_payload("qwen3.6:latest")
        assert payload["model"] == "qwen3.6:latest"
        assert payload["stream"] is False
        assert isinstance(payload["input"], list)
        assert payload["input"][0]["role"] == "user"

    def test_chat_warmup(self):
        backend = LlamaCppBackend().configure(endpoint="chat")
        payload = backend.warmup_payload("qwen3.6:latest")
        assert payload["model"] == "qwen3.6:latest"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "hi"}]


SAMPLE_RENDERED_REQUEST = {
    "custom_id": "row-7",
    "method": "POST",
    "url": "/v1/responses",
    "body": {
        "model": "qwen3.6:latest",
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
        backend = LlamaCppBackend()
        cfg = BatchConfig(model="qwen3.6:latest")
        prep = backend.translate_request(7, SAMPLE_RENDERED_REQUEST, cfg)
        assert prep.index == 7
        assert prep.custom_id == "row-7"
        assert prep.payload == SAMPLE_RENDERED_REQUEST["body"]

    def test_chat_translation(self):
        backend = LlamaCppBackend().configure(endpoint="chat")
        cfg = BatchConfig(model="qwen3.6:latest")
        prep = backend.translate_request(7, SAMPLE_RENDERED_REQUEST, cfg)
        assert prep.index == 7
        assert prep.custom_id == "row-7"

        payload = prep.payload
        assert payload["model"] == "qwen3.6:latest"
        assert payload["stream"] is False
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Classify the email as spam or ham.",
        }
        assert payload["messages"][1] == {"role": "user", "content": "Hello world"}
        assert payload["temperature"] == 0
        assert payload["seed"] == 42
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "spam_result"
        assert payload["response_format"]["json_schema"]["strict"] is True

    def test_chat_translation_uses_config_model_when_missing(self):
        body_no_model = dict(SAMPLE_RENDERED_REQUEST["body"])
        body_no_model.pop("model")
        payload = _responses_body_to_chat_completions(body_no_model, default_model="fallback-model")
        assert payload["model"] == "fallback-model"


class TestResponseWrappers:
    def test_responses_wrap_attaches_custom_id_and_envelope(self):
        resp = {
            "id": "resp_xyz",
            "model": "qwen3.6:latest",
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
        assert body["output"][0]["content"][0]["text"] == '{"label": "spam"}'
        assert body["provider_meta"]["llamacpp"]["model"] == "qwen3.6:latest"
        assert body["provider_meta"]["llamacpp"]["total_tokens"] == 60

    def test_chat_wrap_translates_choices_to_responses_shape(self):
        resp = {
            "id": "chatcmpl_abc",
            "model": "qwen3.6:latest",
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
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == '{"label": "ham"}'
        assert body["usage"]["input_tokens"] == 30
        assert body["usage"]["output_tokens"] == 5

    def test_chat_wrap_handles_empty_choices(self):
        wrapped = _wrap_chat_completion_response("row-x", {"id": "x"})
        body = wrapped["response"]["body"]
        assert body["output"][0]["content"][0]["text"] == ""


class TestBuildSuccessRecord:
    def test_default_uses_responses_wrapper(self):
        backend = LlamaCppBackend()
        resp = {"id": "r1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]}
        rec = backend.build_success_record("cid", resp)
        assert rec["response"]["body"]["output"][0]["content"][0]["text"] == "x"

    def test_chat_kind_uses_chat_wrapper(self):
        backend = LlamaCppBackend().configure(endpoint="chat")
        resp = {
            "id": "c1",
            "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        rec = backend.build_success_record("cid", resp)
        assert rec["response"]["body"]["output"][0]["content"][0]["text"] == "answer"


class TestBackendMetadata:
    def test_name(self):
        assert LlamaCppBackend().name == "llamacpp"

    def test_logger_name(self):
        assert LlamaCppBackend().logger_name == "llm_batch_pipeline.backends.llamacpp"

    def test_batch_id_prefix(self):
        assert LlamaCppBackend().batch_id_prefix == "llamacpp_batch_"
