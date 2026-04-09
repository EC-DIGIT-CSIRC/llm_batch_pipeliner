"""Abstract base class for batch submission backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from llm_batch_pipeline.config import BatchConfig
from llm_batch_pipeline.metrics import MetricsCollector
from llm_batch_pipeline.tui import StatusReporter


@dataclass(slots=True)
class SubmissionResult:
    """Outcome of a batch submission."""

    batch_id: str
    status: str  # completed, failed, expired, cancelled
    output_file: Path | None = None
    error_file: Path | None = None
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


class BatchBackend(ABC):
    """Abstract interface for LLM batch submission backends."""

    @abstractmethod
    def submit(
        self,
        batch_jsonl: Path,
        config: BatchConfig,
        *,
        console: Console | None = None,
        reporter: StatusReporter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> SubmissionResult:
        """Submit a batch JSONL file and return results.

        Implementations may block until completion (Ollama) or poll
        until a terminal state is reached (OpenAI).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. ``'openai'``, ``'ollama'``)."""
