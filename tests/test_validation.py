"""Tests for validation.py — batch output schema validation."""

import textwrap

from llm_batch_pipeline.validation import (
    _extract_output_text,
    validate_batch_output,
)
from tests.conftest import make_batch_output_record, write_batch_output_jsonl


class TestExtractOutputText:
    def test_extracts_from_valid_record(self):
        record = make_batch_output_record("file.eml", '{"label": "spam"}')
        text = _extract_output_text(record)
        assert text == '{"label": "spam"}'

    def test_returns_none_for_missing_output(self):
        record = {"custom_id": "x", "response": {"body": {}}}
        assert _extract_output_text(record) is None

    def test_returns_none_for_missing_response(self):
        record = {"custom_id": "x"}
        assert _extract_output_text(record) is None


class TestValidateBatchOutput:
    def test_no_schema_accepts_valid_json(self, tmp_path):
        records = [
            make_batch_output_record("a.eml", '{"label": "spam"}'),
            make_batch_output_record("b.eml", '{"label": "ham"}'),
        ]
        jsonl = write_batch_output_jsonl(tmp_path, "output.jsonl", records)
        result = validate_batch_output(jsonl)
        assert result.valid_count == 2
        assert result.invalid_count == 0

    def test_with_schema_valid(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="label")
                confidence: float = Field(default=0.0)
        """)
        schema_path = tmp_path / "schema.py"
        schema_path.write_text(schema_code, encoding="utf-8")

        records = [
            make_batch_output_record("a.eml", '{"label": "spam", "confidence": 0.9}'),
        ]
        jsonl = write_batch_output_jsonl(tmp_path, "output.jsonl", records)
        result = validate_batch_output(jsonl, schema_path)
        assert result.valid_count == 1
        assert result.invalid_count == 0

    def test_with_schema_invalid(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, ConfigDict, Field

            class mySchema(BaseModel):
                model_config = ConfigDict(extra="forbid")
                label: str = Field(description="label")
        """)
        schema_path = tmp_path / "schema.py"
        schema_path.write_text(schema_code, encoding="utf-8")

        records = [
            make_batch_output_record("a.eml", '{"label": "spam", "extra_field": "oops"}'),
        ]
        jsonl = write_batch_output_jsonl(tmp_path, "output.jsonl", records)
        result = validate_batch_output(jsonl, schema_path)
        assert result.valid_count == 0
        assert result.invalid_count == 1

    def test_error_records_skipped(self, tmp_path):
        records = [
            make_batch_output_record("a.eml", "", error={"code": "rate_limit"}),
        ]
        jsonl = write_batch_output_jsonl(tmp_path, "output.jsonl", records)
        result = validate_batch_output(jsonl)
        assert result.skipped_count == 1
        assert result.valid_count == 0

    def test_json_parse_error(self, tmp_path):
        records = [
            make_batch_output_record("a.eml", "not valid json {{{"),
        ]
        jsonl = write_batch_output_jsonl(tmp_path, "output.jsonl", records)
        result = validate_batch_output(jsonl)
        assert result.invalid_count == 1
