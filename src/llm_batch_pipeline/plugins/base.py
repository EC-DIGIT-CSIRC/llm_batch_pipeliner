"""Plugin abstract base classes.

Every input-type-specific behaviour is plugged in through these ABCs.
The core pipeline is input-type-agnostic — it only interacts with files
through these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data container passed between stages
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParsedFile:
    """A file after being read and parsed by a :class:`FileReader`.

    Attributes:
        filename:  Original filename (stable identifier / custom_id).
        raw_path:  Absolute path to the source file on disk.
        content:   Plugin-specific parsed representation (opaque to core).
        metadata:  Arbitrary key-value metadata produced during parsing.
    """

    filename: str
    raw_path: Path
    content: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plugin ABCs
# ---------------------------------------------------------------------------


class FileReader(ABC):
    """Reads and parses input files of a specific type.

    Implementations must be stateless and thread-safe.
    """

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Return *True* if this reader can handle *path*."""

    @abstractmethod
    def read(self, path: Path) -> ParsedFile:
        """Read and parse *path*, returning a :class:`ParsedFile`."""

    @abstractmethod
    def package_for_llm(self, parsed: ParsedFile) -> str:
        """Serialize *parsed* into a text string for the LLM user message.

        The returned string is injected verbatim into the ``input_text``
        field of the OpenAI Batch API request body.
        """


class Filter(ABC):
    """Decides whether a parsed file should be kept or dropped.

    Filters are arranged in chains.  Each filter returns a ``(keep, reason)``
    tuple.  If *keep* is ``False`` the file is excluded and *reason* is
    logged.
    """

    @property
    def name(self) -> str:
        """Human-readable filter name for logging."""
        return self.__class__.__name__

    @abstractmethod
    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        """Return ``(keep, reason)``.

        *reason* should be a short human-readable explanation.  When *keep*
        is ``True`` the reason is informational (e.g. ``"passed"``).
        """


class Transformer(ABC):
    """Transforms a :class:`ParsedFile` in-place or returns a new one.

    Transformers are arranged in chains and applied sequentially.
    """

    @property
    def name(self) -> str:
        """Human-readable transformer name for logging."""
        return self.__class__.__name__

    @abstractmethod
    def apply(self, parsed: ParsedFile) -> ParsedFile:
        """Transform *parsed* and return the (possibly new) result."""


class OutputTransformer(ABC):
    """Post-processes validated LLM output rows before evaluation / export.

    Unlike :class:`Transformer` which operates on input files, this works
    on the list of validated result dictionaries coming back from the LLM.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform *rows* and return the modified list."""
