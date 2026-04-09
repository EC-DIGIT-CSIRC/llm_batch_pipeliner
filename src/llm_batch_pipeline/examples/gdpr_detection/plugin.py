"""GDPR / sensitive data detection plugin.

Reads .eml files and detects personally identifiable information (PII)
and other GDPR-relevant data categories.

Registers as plugin name ``gdpr_detection``.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Reuse the email reader from spam detection since the input format is identical
from llm_batch_pipeline.examples.spam_detection.plugin import EmailReader
from llm_batch_pipeline.plugins.base import Filter, ParsedFile, Transformer
from llm_batch_pipeline.plugins.registry import PluginSpec, register_plugin

logger = logging.getLogger("llm_batch_pipeline.examples.gdpr_detection")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class MinLengthFilter(Filter):
    """Drop emails whose body is too short for meaningful PII analysis."""

    def __init__(self, min_chars: int = 20) -> None:
        self._min_chars = min_chars

    @property
    def name(self) -> str:
        return "min_length"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        body = (parsed.content or {}).get("body", "")
        if len(body.strip()) < self._min_chars:
            return False, f"body too short ({len(body.strip())} < {self._min_chars} chars)"
        return True, "sufficient length"


class AutoReplyFilter(Filter):
    """Drop auto-reply / out-of-office messages (unlikely to contain PII)."""

    _AUTO_SUBJECTS = frozenset(
        {
            "out of office",
            "automatic reply",
            "auto-reply",
            "autoreply",
            "delivery status notification",
            "undeliverable",
            "vacation reply",
        }
    )

    @property
    def name(self) -> str:
        return "auto_reply"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        headers = (parsed.content or {}).get("headers", {})
        subject = headers.get("Subject", "").lower()
        for pattern in self._AUTO_SUBJECTS:
            if pattern in subject:
                return False, f"auto-reply detected: {pattern!r}"
        return True, "not auto-reply"


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class RedactAttachmentNamesTransformer(Transformer):
    """Annotate metadata with attachment filenames (for PII in filenames)."""

    @property
    def name(self) -> str:
        return "annotate_attachments"

    def apply(self, parsed: ParsedFile) -> ParsedFile:
        # Already extracted during read; just ensure metadata key exists
        if "attachment_names" not in parsed.metadata:
            parsed.metadata["attachment_names"] = []
        return parsed


# ---------------------------------------------------------------------------
# Custom reader subclass with attachment extraction
# ---------------------------------------------------------------------------


class GdprEmailReader(EmailReader):
    """Extends EmailReader with attachment filename extraction for GDPR."""

    def read(self, path: Path) -> ParsedFile:
        parsed = super().read(path)

        # Extract attachment filenames for PII-in-filename detection
        import email as email_mod  # noqa: PLC0415
        import email.policy  # noqa: PLC0415  # pylint: disable=unused-import  # ensures submodule is loaded

        raw = path.read_bytes()
        msg = email_mod.message_from_bytes(raw, policy=email_mod.policy.default)

        attachments: list[str] = []
        for part in msg.walk():
            fn = part.get_filename()
            if fn:
                attachments.append(fn)

        parsed.metadata["attachment_names"] = attachments
        return parsed

    def package_for_llm(self, parsed: ParsedFile) -> str:
        base_text = super().package_for_llm(parsed)

        attachments = parsed.metadata.get("attachment_names", [])
        if attachments:
            att_section = "\n\nAttachments:\n" + "\n".join(f"  - {a}" for a in attachments)
            return base_text + att_section
        return base_text


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the gdpr_detection plugin."""
    register_plugin(
        PluginSpec(
            name="gdpr_detection",
            reader=GdprEmailReader(),
            pre_filters=[MinLengthFilter(), AutoReplyFilter()],
            transformers=[RedactAttachmentNamesTransformer()],
        )
    )
