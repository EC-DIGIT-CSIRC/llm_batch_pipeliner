"""Tests for transforms.py — transform chain execution."""

from pathlib import Path

from llm_batch_pipeline.plugins.base import ParsedFile, Transformer
from llm_batch_pipeline.transforms import run_transform_chain


class _UpperTransformer(Transformer):
    @property
    def name(self) -> str:
        return "upper"

    def apply(self, parsed: ParsedFile) -> ParsedFile:
        parsed.content = str(parsed.content).upper()
        return parsed


class _PrefixTransformer(Transformer):
    @property
    def name(self) -> str:
        return "prefix"

    def apply(self, parsed: ParsedFile) -> ParsedFile:
        parsed.content = "PREFIX:" + str(parsed.content)
        return parsed


class _FailingTransformer(Transformer):
    @property
    def name(self) -> str:
        return "failing"

    def apply(self, parsed: ParsedFile) -> ParsedFile:
        msg = "intentional failure"
        raise RuntimeError(msg)


def _make_file(name: str, content: str) -> ParsedFile:
    return ParsedFile(filename=name, raw_path=Path(f"/tmp/{name}"), content=content)


class TestRunTransformChain:
    def test_empty_transformers(self):
        files = [_make_file("a.txt", "hello")]
        result = run_transform_chain(files, [])
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_single_transformer(self):
        files = [_make_file("a.txt", "hello")]
        result = run_transform_chain(files, [_UpperTransformer()])
        assert result[0].content == "HELLO"

    def test_chain_applies_sequentially(self):
        files = [_make_file("a.txt", "hello")]
        result = run_transform_chain(files, [_UpperTransformer(), _PrefixTransformer()])
        assert result[0].content == "PREFIX:HELLO"

    def test_failing_transformer_passes_through(self):
        files = [_make_file("a.txt", "hello")]
        result = run_transform_chain(files, [_FailingTransformer()])
        assert len(result) == 1
        # File passes through unchanged
        assert result[0].content == "hello"

    def test_multiple_files(self):
        files = [_make_file(f"{i}.txt", f"content_{i}") for i in range(3)]
        result = run_transform_chain(files, [_UpperTransformer()])
        assert len(result) == 3
        assert all("CONTENT_" in str(r.content) for r in result)
