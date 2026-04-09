"""Tests for filters.py — filter chain execution."""

from pathlib import Path

from llm_batch_pipeline.filters import run_filter_chain
from llm_batch_pipeline.plugins.base import Filter, ParsedFile


class _KeepAllFilter(Filter):
    @property
    def name(self) -> str:
        return "keep_all"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        return True, "always keep"


class _DropShortFilter(Filter):
    @property
    def name(self) -> str:
        return "drop_short"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        text = str(parsed.content or "")
        if len(text) < 10:
            return False, f"too short ({len(text)} chars)"
        return True, "passed"


class _DropAllFilter(Filter):
    @property
    def name(self) -> str:
        return "drop_all"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        return False, "dropped by policy"


def _make_file(name: str, content: str) -> ParsedFile:
    return ParsedFile(filename=name, raw_path=Path(f"/tmp/{name}"), content=content)


class TestRunFilterChain:
    def test_empty_filters(self):
        files = [_make_file("a.txt", "hello")]
        result = run_filter_chain(files, [])
        assert result.kept_count == 1
        assert result.dropped_count == 0

    def test_keep_all(self):
        files = [_make_file("a.txt", "hello world")]
        result = run_filter_chain(files, [_KeepAllFilter()])
        assert result.kept_count == 1
        assert result.dropped_count == 0

    def test_drop_short(self):
        files = [
            _make_file("long.txt", "this is long enough"),
            _make_file("short.txt", "hi"),
        ]
        result = run_filter_chain(files, [_DropShortFilter()])
        assert result.kept_count == 1
        assert result.dropped_count == 1
        assert result.dropped[0]["filename"] == "short.txt"
        assert result.dropped[0]["filter"] == "drop_short"

    def test_drop_all(self):
        files = [_make_file("a.txt", "hello"), _make_file("b.txt", "world")]
        result = run_filter_chain(files, [_DropAllFilter()])
        assert result.kept_count == 0
        assert result.dropped_count == 2

    def test_chain_order_first_rejection_wins(self):
        files = [_make_file("x.txt", "hi")]
        result = run_filter_chain(files, [_DropShortFilter(), _DropAllFilter()])
        assert result.dropped_count == 1
        assert result.dropped[0]["filter"] == "drop_short"

    def test_total_input_tracked(self):
        files = [_make_file(f"{i}.txt", "content" * i) for i in range(5)]
        result = run_filter_chain(files, [_DropShortFilter()])
        assert result.total_input == 5
