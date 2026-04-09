"""Tests for plugins/registry.py — plugin registration and discovery."""

from pathlib import Path

import pytest

from llm_batch_pipeline.plugins.base import FileReader, ParsedFile
from llm_batch_pipeline.plugins.registry import (
    _REGISTRY,
    PluginSpec,
    get_plugin,
    list_plugins,
    register_plugin,
)


class _DummyReader(FileReader):
    def can_read(self, path: Path) -> bool:
        return True

    def read(self, path: Path) -> ParsedFile:
        return ParsedFile(filename=path.name, raw_path=path)

    def package_for_llm(self, parsed: ParsedFile) -> str:
        return ""


class TestRegistry:
    def test_register_and_get(self):
        spec = PluginSpec(name="test_plugin_reg", reader=_DummyReader())
        register_plugin(spec)
        try:
            result = get_plugin("test_plugin_reg")
            assert result.name == "test_plugin_reg"
        finally:
            _REGISTRY.pop("test_plugin_reg", None)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown plugin"):
            get_plugin("nonexistent_plugin_xyz_42")

    def test_list_plugins_includes_builtins(self):
        plugins = list_plugins()
        assert "spam_detection" in plugins
        assert "gdpr_detection" in plugins

    def test_overwrite_registration(self):
        spec1 = PluginSpec(name="test_overwrite", reader=_DummyReader())
        spec2 = PluginSpec(name="test_overwrite", reader=_DummyReader())
        register_plugin(spec1)
        register_plugin(spec2)
        try:
            result = get_plugin("test_overwrite")
            assert result is spec2  # Last one wins
        finally:
            _REGISTRY.pop("test_overwrite", None)
