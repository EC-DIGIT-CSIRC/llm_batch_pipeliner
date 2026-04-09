"""Tests for the spam_detection plugin."""

from pathlib import Path

from llm_batch_pipeline.examples.spam_detection.plugin import (
    EmailReader,
    EmptyBodyFilter,
    TrimWhitespaceTransformer,
)
from llm_batch_pipeline.plugins.registry import get_plugin
from tests.conftest import (
    EMPTY_BODY_EML,
    HTML_EMAIL,
    MULTIPART_EML,
    SIMPLE_HAM_EML,
    write_eml,
)


class TestEmailReader:
    def test_can_read_eml(self, tmp_path):
        reader = EmailReader()
        assert reader.can_read(Path("test.eml"))
        assert reader.can_read(Path("test.txt"))
        assert not reader.can_read(Path("test.csv"))
        assert not reader.can_read(Path("test.py"))

    def test_read_simple_email(self, tmp_path):
        path = write_eml(tmp_path, "ham.eml", SIMPLE_HAM_EML)
        reader = EmailReader()
        parsed = reader.read(path)

        assert parsed.filename == "ham.eml"
        assert parsed.content["headers"]["From"] == "alice@example.com"
        assert parsed.content["headers"]["Subject"] == "Meeting tomorrow"
        assert "meet tomorrow" in parsed.content["body"].lower()

    def test_read_html_email(self, tmp_path):
        path = write_eml(tmp_path, "html.eml", HTML_EMAIL)
        reader = EmailReader()
        parsed = reader.read(path)

        body = parsed.content["body"]
        assert "newsletter" in body.lower()
        # HTML tags should be stripped
        assert "<html>" not in body

    def test_read_multipart(self, tmp_path):
        path = write_eml(tmp_path, "multi.eml", MULTIPART_EML)
        reader = EmailReader()
        parsed = reader.read(path)

        # Should prefer plain text
        assert "plain text version" in parsed.content["body"].lower()

    def test_package_for_llm(self, tmp_path):
        path = write_eml(tmp_path, "test.eml", SIMPLE_HAM_EML)
        reader = EmailReader()
        parsed = reader.read(path)
        llm_text = reader.package_for_llm(parsed)

        assert "From:" in llm_text
        assert "alice@example.com" in llm_text
        assert "Meeting tomorrow" in llm_text

    def test_empty_email(self, tmp_path):
        path = write_eml(tmp_path, "empty.eml", EMPTY_BODY_EML)
        reader = EmailReader()
        parsed = reader.read(path)
        assert parsed.content["body"] == ""


class TestEmptyBodyFilter:
    def test_keeps_normal_email(self, tmp_path):
        path = write_eml(tmp_path, "ham.eml", SIMPLE_HAM_EML)
        reader = EmailReader()
        parsed = reader.read(path)
        flt = EmptyBodyFilter()
        keep, reason = flt.apply(parsed)
        assert keep

    def test_drops_empty_email(self, tmp_path):
        path = write_eml(tmp_path, "empty.eml", EMPTY_BODY_EML)
        reader = EmailReader()
        parsed = reader.read(path)
        flt = EmptyBodyFilter()
        keep, reason = flt.apply(parsed)
        assert not keep
        assert "empty" in reason.lower() or "short" in reason.lower()


class TestTrimWhitespaceTransformer:
    def test_collapses_whitespace(self, tmp_path):
        path = write_eml(tmp_path, "test.eml", SIMPLE_HAM_EML)
        reader = EmailReader()
        parsed = reader.read(path)
        # Inject excessive whitespace
        parsed.content["body"] = "line1\n\n\n\n\nline2\n\n\n\nline3"

        tx = TrimWhitespaceTransformer()
        result = tx.apply(parsed)
        body = result.content["body"]
        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in body


class TestPluginRegistration:
    def test_auto_discovery(self):
        plugins = get_plugin("spam_detection")
        assert plugins.name == "spam_detection"
        assert isinstance(plugins.reader, EmailReader)
