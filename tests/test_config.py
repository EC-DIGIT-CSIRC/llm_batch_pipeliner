"""Tests for config.py — BatchConfig, TOML loading, batch dir helpers."""

from pathlib import Path

import pytest

from llm_batch_pipeline.config import (
    BatchConfig,
    apply_toml_overrides,
    create_batch_dir,
    load_batch_toml,
    next_batch_number,
    resolve_batch_dir,
)


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.batch_dir == Path(".")
        assert cfg.input_dir == Path("input")
        assert cfg.model == "gpt-4o-mini"
        assert cfg.backend == "openai"
        assert cfg.auto_approve is False

    def test_custom_input_dir(self, tmp_path):
        custom = tmp_path / "my_input"
        cfg = BatchConfig(input_dir=custom)
        assert cfg.input_dir == custom

    def test_default_input_dir_derived_from_batch_dir(self, tmp_path):
        cfg = BatchConfig(batch_dir=tmp_path)
        assert cfg.input_dir == tmp_path / "input"

    def test_derived_paths(self, tmp_path):
        cfg = BatchConfig(batch_dir=tmp_path)
        assert cfg.job_dir == tmp_path / "job"
        assert cfg.output_dir == tmp_path / "output"
        assert cfg.results_dir == tmp_path / "results"
        assert cfg.export_dir == tmp_path / "export"
        assert cfg.logs_dir == tmp_path / "logs"
        assert cfg.evaluation_dir == tmp_path / "evaluation"


class TestTomlLoading:
    def test_load_nonexistent(self, tmp_path):
        result = load_batch_toml(tmp_path / "does_not_exist.toml")
        assert result == {}

    def test_load_valid_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('model = "llama3"\nbackend = "ollama"\n', encoding="utf-8")
        result = load_batch_toml(toml_path)
        assert result["model"] == "llama3"
        assert result["backend"] == "ollama"

    def test_apply_overrides(self):
        cfg = BatchConfig()
        apply_toml_overrides(cfg, {"model": "gpt-4o", "backend": "ollama", "unknown_key": "ignored"})
        assert cfg.model == "gpt-4o"
        assert cfg.backend == "ollama"


class TestBatchDirHelpers:
    def test_next_batch_number_empty(self, tmp_path):
        assert next_batch_number(tmp_path) == 1

    def test_next_batch_number_with_existing(self, tmp_path):
        (tmp_path / "batch_001_test").mkdir()
        (tmp_path / "batch_003_other").mkdir()
        assert next_batch_number(tmp_path) == 4

    def test_create_batch_dir(self, tmp_path):
        batch_dir = create_batch_dir(tmp_path, "spam_run")
        assert batch_dir.name == "batch_001_spam_run"
        assert (batch_dir / "input").is_dir()
        assert (batch_dir / "evaluation").is_dir()

    def test_create_batch_dir_sequential(self, tmp_path):
        create_batch_dir(tmp_path, "first")
        second = create_batch_dir(tmp_path, "second")
        assert second.name == "batch_002_second"

    def test_resolve_batch_dir_by_path(self, tmp_path):
        batch_dir = tmp_path / "my_batch"
        batch_dir.mkdir()
        resolved = resolve_batch_dir(tmp_path, str(batch_dir))
        assert resolved == batch_dir.resolve()

    def test_resolve_batch_dir_by_name(self, tmp_path):
        batch_dir = tmp_path / "batch_001_test"
        batch_dir.mkdir()
        resolved = resolve_batch_dir(tmp_path, "test")
        assert resolved == batch_dir.resolve()

    def test_resolve_batch_dir_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_batch_dir(tmp_path, "nonexistent")
