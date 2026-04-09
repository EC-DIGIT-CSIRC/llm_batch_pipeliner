"""Tests for render.py — batch JSONL rendering and sharding."""

import json
from pathlib import Path

from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.plugins.base import FileReader, ParsedFile
from llm_batch_pipeline.render import render_batch


class _SimpleReader(FileReader):
    """Test reader that returns content as-is for LLM packaging."""

    def can_read(self, path: Path) -> bool:
        return True

    def read(self, path: Path) -> ParsedFile:
        return ParsedFile(filename=path.name, raw_path=path, content=path.read_text())

    def package_for_llm(self, parsed: ParsedFile) -> str:
        return str(parsed.content)


class TestRenderBatch:
    def _make_config(self, tmp_path: Path, **overrides) -> BatchConfig:
        defaults: dict = {
            "batch_dir": tmp_path,
            "model": "test-model",
            "prompt_file": None,
            "schema_file": None,
        }
        defaults.update(overrides)
        return BatchConfig(**defaults)

    def test_single_file_single_shard(self, tmp_path):
        # Write a prompt
        (tmp_path / "prompt.txt").write_text("Analyse this content.", encoding="utf-8")

        files = [
            ParsedFile(filename="test.eml", raw_path=Path("/tmp/test.eml"), content="Hello world"),
        ]
        reader = _SimpleReader()
        config = self._make_config(tmp_path)

        shard_paths = render_batch(files, reader, config)
        assert len(shard_paths) == 1

        # Verify JSONL content
        content = shard_paths[0].read_text(encoding="utf-8")
        records = [json.loads(line) for line in content.strip().splitlines()]
        assert len(records) == 1
        assert records[0]["custom_id"] == "test.eml"
        assert records[0]["method"] == "POST"
        assert records[0]["url"] == "/v1/responses"
        assert records[0]["body"]["model"] == "test-model"
        assert records[0]["body"]["instructions"] == "Analyse this content."

    def test_multiple_files(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("Classify.", encoding="utf-8")

        files = [
            ParsedFile(filename=f"file_{i}.eml", raw_path=Path(f"/tmp/file_{i}.eml"), content=f"Content {i}")
            for i in range(5)
        ]
        reader = _SimpleReader()
        config = self._make_config(tmp_path)

        shard_paths = render_batch(files, reader, config)
        assert len(shard_paths) == 1

        content = shard_paths[0].read_text(encoding="utf-8")
        records = [json.loads(line) for line in content.strip().splitlines()]
        assert len(records) == 5
        assert {r["custom_id"] for r in records} == {f"file_{i}.eml" for i in range(5)}

    def test_sharding_by_request_count(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("Go.", encoding="utf-8")

        files = [ParsedFile(filename=f"f{i}.eml", raw_path=Path(f"/tmp/f{i}.eml"), content=f"C{i}") for i in range(10)]
        reader = _SimpleReader()
        config = self._make_config(tmp_path, max_requests_per_shard=3)

        shard_paths = render_batch(files, reader, config)
        assert len(shard_paths) == 4  # 3+3+3+1

    def test_symlink_for_single_shard(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("Test.", encoding="utf-8")

        files = [ParsedFile(filename="a.eml", raw_path=Path("/tmp/a.eml"), content="content")]
        reader = _SimpleReader()
        config = self._make_config(tmp_path)

        render_batch(files, reader, config)
        symlink = config.job_dir / "batch.jsonl"
        assert symlink.is_symlink()

    def test_with_schema(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("Analyse.", encoding="utf-8")
        schema_code = (
            "from pydantic import BaseModel, Field\n\n"
            "class mySchema(BaseModel):\n"
            '    label: str = Field(description="test")\n'
        )
        schema_path = tmp_path / "schema.py"
        schema_path.write_text(schema_code, encoding="utf-8")

        files = [ParsedFile(filename="a.eml", raw_path=Path("/tmp/a.eml"), content="content")]
        reader = _SimpleReader()
        config = self._make_config(tmp_path, schema_file=schema_path)

        shard_paths = render_batch(files, reader, config)
        content = shard_paths[0].read_text(encoding="utf-8")
        record = json.loads(content.strip())
        assert "text" in record["body"]
        assert record["body"]["text"]["format"]["type"] == "json_schema"

    def test_default_prompt(self, tmp_path):
        """When no prompt file exists, uses the default."""
        files = [ParsedFile(filename="a.eml", raw_path=Path("/tmp/a.eml"), content="content")]
        reader = _SimpleReader()
        config = self._make_config(tmp_path)

        shard_paths = render_batch(files, reader, config)
        content = shard_paths[0].read_text(encoding="utf-8")
        record = json.loads(content.strip())
        assert "schema" in record["body"]["instructions"].lower() or len(record["body"]["instructions"]) > 0
