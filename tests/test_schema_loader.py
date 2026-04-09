"""Tests for schema_loader.py — dynamic schema loading and strict enforcement."""

import textwrap
from pathlib import Path

import pytest

from llm_batch_pipeline.schema_loader import (
    _ensure_strict_json_schema,
    infer_confidence_field,
    infer_label_field,
    load_schema_class,
    load_schema_format,
)


def _write_schema(tmp_path: Path, content: str) -> Path:
    """Write a schema file and return its path."""
    p = tmp_path / "test_schema.py"
    p.write_text(content, encoding="utf-8")
    return p


class TestLoadSchemaClass:
    def test_loads_myschema(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="test")
        """)
        path = _write_schema(tmp_path, schema_code)
        cls = load_schema_class(path)
        assert cls.__name__ == "mySchema"
        assert "label" in cls.model_fields

    def test_missing_myschema_raises(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel

            class OtherSchema(BaseModel):
                x: int = 0
        """)
        path = _write_schema(tmp_path, schema_code)
        with pytest.raises(ValueError, match="must define.*mySchema"):
            load_schema_class(path)


class TestLoadSchemaFormat:
    def test_returns_text_format(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="test")
                score: float = Field(default=0.0)
        """)
        path = _write_schema(tmp_path, schema_code)
        fmt = load_schema_format(path)
        assert "format" in fmt
        assert fmt["format"]["type"] == "json_schema"
        assert fmt["format"]["strict"] is True
        assert "schema" in fmt["format"]


class TestEnsureStrictJsonSchema:
    def test_adds_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        _ensure_strict_json_schema(schema)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"name", "age"}

    def test_recurses_into_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                    },
                }
            },
        }
        _ensure_strict_json_schema(schema)
        assert schema["properties"]["nested"]["additionalProperties"] is False
        assert schema["properties"]["nested"]["required"] == ["inner"]

    def test_handles_defs(self):
        schema = {
            "type": "object",
            "properties": {},
            "$defs": {
                "Sub": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                }
            },
        }
        _ensure_strict_json_schema(schema)
        assert schema["$defs"]["Sub"]["additionalProperties"] is False


class TestFieldInference:
    def test_infer_label_from_enum(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from typing import Literal
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                classification: Literal["spam", "ham"] = Field(description="label")
                confidence: float = Field(default=0.0)
        """)
        path = _write_schema(tmp_path, schema_code)
        assert infer_label_field(path) == "classification"

    def test_infer_label_from_name(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="label")
        """)
        path = _write_schema(tmp_path, schema_code)
        assert infer_label_field(path) == "label"

    def test_infer_confidence(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="label")
                confidence: float = Field(default=0.0)
        """)
        path = _write_schema(tmp_path, schema_code)
        assert infer_confidence_field(path) == "confidence"

    def test_infer_no_confidence_returns_none(self, tmp_path):
        schema_code = textwrap.dedent("""\
            from pydantic import BaseModel, Field

            class mySchema(BaseModel):
                label: str = Field(description="label")
                reason: str = Field(default="")
        """)
        path = _write_schema(tmp_path, schema_code)
        assert infer_confidence_field(path) is None
