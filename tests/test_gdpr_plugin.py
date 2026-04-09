"""Tests for the gdpr_detection plugin."""

from pathlib import Path

from llm_batch_pipeline.examples.gdpr_detection.plugin import (
    AutoReplyFilter,
    GdprEmailReader,
    MinLengthFilter,
)
from llm_batch_pipeline.plugins.registry import get_plugin
from tests.conftest import (
    AUTO_REPLY_EML,
    PII_EMAIL,
    SIMPLE_HAM_EML,
    write_eml,
)


class TestGdprEmailReader:
    def test_read_with_attachments_metadata(self, tmp_path):
        path = write_eml(tmp_path, "pii.eml", PII_EMAIL)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        assert "attachment_names" in parsed.metadata
        # Simple text email has no attachments
        assert parsed.metadata["attachment_names"] == []

    def test_package_for_llm_no_attachments(self, tmp_path):
        path = write_eml(tmp_path, "test.eml", SIMPLE_HAM_EML)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        text = reader.package_for_llm(parsed)
        assert "Attachments:" not in text

    def test_inherits_email_reader(self, tmp_path):
        path = write_eml(tmp_path, "test.eml", SIMPLE_HAM_EML)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        assert parsed.content["headers"]["From"] == "alice@example.com"


class TestMinLengthFilter:
    def test_keeps_long_email(self, tmp_path):
        path = write_eml(tmp_path, "long.eml", PII_EMAIL)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        flt = MinLengthFilter()
        keep, reason = flt.apply(parsed)
        assert keep

    def test_drops_short_email(self, tmp_path):
        from llm_batch_pipeline.plugins.base import ParsedFile

        pf = ParsedFile(filename="short.eml", raw_path=Path("/tmp/short.eml"), content={"body": "hi"})
        flt = MinLengthFilter()
        keep, reason = flt.apply(pf)
        assert not keep
        assert "short" in reason.lower()


class TestAutoReplyFilter:
    def test_drops_auto_reply(self, tmp_path):
        path = write_eml(tmp_path, "autoreply.eml", AUTO_REPLY_EML)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        flt = AutoReplyFilter()
        keep, reason = flt.apply(parsed)
        assert not keep
        assert "auto-reply" in reason.lower() or "out of office" in reason.lower()

    def test_keeps_normal_email(self, tmp_path):
        path = write_eml(tmp_path, "normal.eml", SIMPLE_HAM_EML)
        reader = GdprEmailReader()
        parsed = reader.read(path)
        flt = AutoReplyFilter()
        keep, reason = flt.apply(parsed)
        assert keep


class TestPluginRegistration:
    def test_auto_discovery(self):
        plugin = get_plugin("gdpr_detection")
        assert plugin.name == "gdpr_detection"
        assert isinstance(plugin.reader, GdprEmailReader)
        assert len(plugin.pre_filters) == 2
