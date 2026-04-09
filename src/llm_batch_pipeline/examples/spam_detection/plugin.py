"""Spam detection plugin: reads raw .eml files for spam/ham classification.

Registers as plugin name ``spam_detection``.
"""

from __future__ import annotations

import email
import email.header
import email.policy
import logging
from pathlib import Path
from typing import Any

from llm_batch_pipeline.plugins.base import FileReader, Filter, ParsedFile, Transformer
from llm_batch_pipeline.plugins.registry import PluginSpec, register_plugin

logger = logging.getLogger("llm_batch_pipeline.examples.spam_detection")


# ---------------------------------------------------------------------------
# FileReader
# ---------------------------------------------------------------------------

_EML_EXTENSIONS = frozenset({".eml", ".txt", ".msg"})

# Non-email extensions we should NOT try to parse
_SKIP_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".zip",
        ".gz",
        ".tar",
        ".bz2",
        ".7z",
        ".rar",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".py",
        ".js",
        ".ts",
        ".css",
        ".html",
        ".json",
        ".xml",
        ".csv",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".conf",
        ".md",
        ".rst",
        ".log",
    }
)


class EmailReader(FileReader):
    """Reads RFC-822 email files and extracts headers + visible body text."""

    def can_read(self, path: Path) -> bool:
        # Accept known email extensions
        suffix = path.suffix.lower()
        if suffix in _EML_EXTENSIONS:
            return True
        # Reject known non-email extensions; accept extensionless files
        # or files with hash-like suffixes
        # (e.g. "easy_ham__00027.4d456dd9ce0afde7629f94dc3034e0bb")
        return suffix not in _SKIP_EXTENSIONS

    def read(self, path: Path) -> ParsedFile:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw, policy=email.policy.default)

        headers = _extract_headers(msg)
        body = _extract_body_text(msg)
        defects = [str(d) for d in msg.defects] if msg.defects else []

        return ParsedFile(
            filename=path.name,
            raw_path=path.resolve(),
            content={"headers": headers, "body": body},
            metadata={
                "defects": defects,
                "content_type": msg.get_content_type(),
                "num_parts": sum(1 for _ in msg.walk()),
            },
        )

    def package_for_llm(self, parsed: ParsedFile) -> str:
        content = parsed.content or {}
        headers = content.get("headers", {})
        body = content.get("body", "")

        lines: list[str] = []
        for key in ("From", "To", "Subject", "Date", "Reply-To"):
            val = headers.get(key)
            if val:
                lines.append(f"{key}: {val}")
        lines.append("")
        lines.append(body[:30_000])  # Cap body to avoid token overflows
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class EmptyBodyFilter(Filter):
    """Drop emails with no usable body text."""

    @property
    def name(self) -> str:
        return "empty_body"

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        body = (parsed.content or {}).get("body", "")
        if not body or len(body.strip()) < 10:
            return False, "body is empty or too short (<10 chars)"
        return True, "body present"


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class TrimWhitespaceTransformer(Transformer):
    """Normalize excessive whitespace in body text."""

    @property
    def name(self) -> str:
        return "trim_whitespace"

    def apply(self, parsed: ParsedFile) -> ParsedFile:
        if parsed.content and "body" in parsed.content:
            import re  # noqa: PLC0415

            text = parsed.content["body"]
            # Collapse runs of 3+ newlines to 2
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Collapse runs of spaces/tabs (not newlines) to single space
            text = re.sub(r"[^\S\n]+", " ", text)
            parsed.content["body"] = text.strip()
        return parsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_headers(msg: email.message.Message) -> dict[str, str]:
    """Extract key headers as plain strings."""
    result: dict[str, str] = {}
    for key in ("From", "To", "Cc", "Subject", "Date", "Reply-To", "Return-Path", "Message-ID"):
        value = msg.get(key)
        if value:
            # Decode RFC2047 encoded headers
            decoded = _decode_header(value)
            result[key] = decoded
    return result


def _decode_header(value: Any) -> str:
    """Decode an email header value, handling RFC 2047 encoding."""
    if isinstance(value, str):
        return value
    if hasattr(value, "__str__"):
        return str(value)
    return str(value)


def _extract_body_text(msg: email.message.Message) -> str:
    """Walk MIME tree and extract the best plain-text body.

    Prefers text/plain. Falls back to text/html with basic tag stripping.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "text/plain":
            text = _get_text_payload(part)
            if text:
                plain_parts.append(text)
        elif ct == "text/html":
            text = _get_text_payload(part)
            if text:
                html_parts.append(text)

    if plain_parts:
        return "\n\n".join(plain_parts)

    if html_parts:
        return "\n\n".join(_strip_html(h) for h in html_parts)

    return ""


def _get_text_payload(part: email.message.Message) -> str:
    """Safely extract text payload from a MIME part."""
    try:
        payload = part.get_content()
        if isinstance(payload, str):
            return payload
        if isinstance(payload, bytes):
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                return payload.decode("utf-8", errors="replace")
    except (KeyError, LookupError, UnicodeDecodeError):
        pass
    return ""


def _strip_html(html: str) -> str:
    """Basic HTML tag stripping for fallback when selectolax is unavailable."""
    try:
        from selectolax.parser import HTMLParser  # noqa: PLC0415

        tree = HTMLParser(html)
        # Remove script and style elements
        for tag in tree.css("script, style, head"):
            tag.decompose()
        text = tree.text(separator="\n")
        return text.strip() if text else ""
    except ImportError:
        import re  # noqa: PLC0415

        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the spam_detection plugin."""
    register_plugin(
        PluginSpec(
            name="spam_detection",
            reader=EmailReader(),
            pre_filters=[EmptyBodyFilter()],
            transformers=[TrimWhitespaceTransformer()],
        )
    )
