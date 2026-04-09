"""Tests for plugins/base.py — ParsedFile and ABC contracts."""

from pathlib import Path

from llm_batch_pipeline.plugins.base import (
    FileReader,
    Filter,
    OutputTransformer,
    ParsedFile,
    Transformer,
)


class TestParsedFile:
    def test_creation(self):
        pf = ParsedFile(filename="test.eml", raw_path=Path("/tmp/test.eml"))
        assert pf.filename == "test.eml"
        assert pf.content is None
        assert pf.metadata == {}

    def test_with_content(self):
        pf = ParsedFile(
            filename="test.eml",
            raw_path=Path("/tmp/test.eml"),
            content={"body": "hello"},
            metadata={"size": 100},
        )
        assert pf.content["body"] == "hello"
        assert pf.metadata["size"] == 100


class TestABCContracts:
    """Verify ABC subclass contracts work correctly."""

    def test_file_reader_subclass(self):
        class TestReader(FileReader):
            def can_read(self, path: Path) -> bool:
                return path.suffix == ".txt"

            def read(self, path: Path) -> ParsedFile:
                return ParsedFile(filename=path.name, raw_path=path, content=path.read_text())

            def package_for_llm(self, parsed: ParsedFile) -> str:
                return str(parsed.content)

        reader = TestReader()
        assert reader.can_read(Path("file.txt"))
        assert not reader.can_read(Path("file.csv"))

    def test_filter_subclass(self):
        class TestFilter(Filter):
            def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
                return len(str(parsed.content)) > 5, "length check"

        f = TestFilter()
        assert f.name == "TestFilter"
        keep, reason = f.apply(ParsedFile(filename="x", raw_path=Path("."), content="long enough"))
        assert keep
        keep, reason = f.apply(ParsedFile(filename="x", raw_path=Path("."), content="hi"))
        assert not keep

    def test_transformer_subclass(self):
        class TestTransformer(Transformer):
            def apply(self, parsed: ParsedFile) -> ParsedFile:
                parsed.content = str(parsed.content).upper()
                return parsed

        t = TestTransformer()
        pf = ParsedFile(filename="x", raw_path=Path("."), content="hello")
        result = t.apply(pf)
        assert result.content == "HELLO"

    def test_output_transformer_subclass(self):
        class TestOutputTransformer(OutputTransformer):
            def apply(self, rows: list[dict]) -> list[dict]:
                return [dict(r, processed=True) for r in rows]

        t = TestOutputTransformer()
        result = t.apply([{"a": 1}])
        assert result[0]["processed"] is True
