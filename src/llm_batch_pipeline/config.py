"""Configuration dataclasses and defaults.

Each pipeline stage has its own config dataclass.  A top-level
:class:`BatchConfig` bundles per-batch settings and is built from CLI
arguments, per-batch ``config.toml``, and environment variables (in that
precedence order).
"""

from __future__ import annotations

import contextlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENAI_BATCH_ENDPOINT: str = "/v1/responses"
OLLAMA_CHAT_ENDPOINT: str = "/api/chat"

DEFAULT_BATCH_MAX_REQUESTS: int = 50_000
DEFAULT_BATCH_MAX_BYTES: int = 190 * 1024 * 1024  # 190 MiB

DEFAULT_BATCH_POLL_INTERVAL_SECONDS: int = 15
DEFAULT_OLLAMA_NUM_PARALLEL_JOBS: int = 3
DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS: int = 600

DEFAULT_METRICS_PORT: int = 9090

BATCH_DIR_PREFIX: str = "batch_"


# ---------------------------------------------------------------------------
# Per-batch configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchConfig:
    """Top-level configuration for a single batch job run."""

    # Identity
    batch_dir: Path = field(default_factory=lambda: Path("."))
    batch_name: str = ""

    # Input
    input_dir: Path | None = None
    prompt_file: Path | None = None
    schema_file: Path | None = None

    # Plugin
    plugin_name: str = ""

    # Rendering
    model: str = "gpt-4o-mini"
    max_requests_per_shard: int = DEFAULT_BATCH_MAX_REQUESTS
    max_bytes_per_shard: int = DEFAULT_BATCH_MAX_BYTES

    # Backend
    backend: str = "openai"  # "openai" | "ollama"
    base_urls: list[str] = field(default_factory=lambda: ["http://localhost:11434"])
    num_shards: int | None = None
    num_parallel_jobs: int = DEFAULT_OLLAMA_NUM_PARALLEL_JOBS
    request_timeout_seconds: int = DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS
    poll_interval_seconds: int = DEFAULT_BATCH_POLL_INTERVAL_SECONDS
    completion_window: str = "24h"
    insecure: bool = False
    no_wait: bool = False

    # Prompt override (submit-time)
    prompt_override: str | None = None
    prompt_override_file: Path | None = None

    # Evaluation
    ground_truth_csv: Path | None = None
    category_map_file: Path | None = None
    label_field: str | None = None
    confidence_field: str | None = None
    positive_class: str | None = None

    # Observability
    metrics_port: int | None = None  # None = metrics HTTP server disabled
    log_level: str = "INFO"

    # Workflow
    auto_approve: bool = False
    start_from: str | None = None
    dry_run: bool = False

    # Resume
    resume_batch_id: str | None = None

    def __post_init__(self) -> None:
        self.batch_dir = Path(self.batch_dir)
        if self.input_dir is None:
            self.input_dir = self.batch_dir / "input"
        else:
            self.input_dir = Path(self.input_dir)

    # Derived paths ----------------------------------------------------------

    @property
    def job_dir(self) -> Path:
        return self.batch_dir / "job"

    @property
    def output_dir(self) -> Path:
        return self.batch_dir / "output"

    @property
    def results_dir(self) -> Path:
        return self.batch_dir / "results"

    @property
    def export_dir(self) -> Path:
        return self.batch_dir / "export"

    @property
    def logs_dir(self) -> Path:
        return self.batch_dir / "logs"

    @property
    def evaluation_dir(self) -> Path:
        return self.batch_dir / "evaluation"


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------


def load_batch_toml(path: Path) -> dict[str, Any]:
    """Load a per-batch ``config.toml`` and return as a plain dict.

    Returns an empty dict when *path* does not exist.
    """
    if not path.is_file():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def apply_toml_overrides(config: BatchConfig, overrides: dict[str, Any]) -> None:
    """Merge *overrides* from a TOML file into *config*.

    Only known fields are applied; unknown keys are silently ignored.
    """
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)


# ---------------------------------------------------------------------------
# Helper: resolve batch directory
# ---------------------------------------------------------------------------


def resolve_batch_dir(batch_jobs_root: Path, name_or_path: str) -> Path:
    """Resolve a batch job directory from a name or path.

    If *name_or_path* is an existing directory, use it directly.
    Otherwise search *batch_jobs_root* for a directory whose name ends
    with ``_<name_or_path>`` or equals *name_or_path*.
    """
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate.resolve()

    # Search batch_jobs_root
    if batch_jobs_root.is_dir():
        for child in sorted(batch_jobs_root.iterdir()):
            if child.is_dir() and (child.name == name_or_path or child.name.endswith(f"_{name_or_path}")):
                return child.resolve()

    msg = f"Batch directory not found: {name_or_path} (searched {batch_jobs_root})"
    raise FileNotFoundError(msg)


def next_batch_number(batch_jobs_root: Path) -> int:
    """Return the next available batch sequence number."""
    max_num = 0
    if batch_jobs_root.is_dir():
        for child in batch_jobs_root.iterdir():
            if child.is_dir() and child.name.startswith(BATCH_DIR_PREFIX):
                # Extract NNN from batch_NNN_name
                parts = child.name[len(BATCH_DIR_PREFIX) :].split("_", 1)
                with contextlib.suppress(ValueError, IndexError):
                    max_num = max(max_num, int(parts[0]))
    return max_num + 1


def create_batch_dir(batch_jobs_root: Path, name: str) -> Path:
    """Create a new batch job directory with auto-numbered prefix.

    Returns the path to the created directory.
    """
    num = next_batch_number(batch_jobs_root)
    dir_name = f"{BATCH_DIR_PREFIX}{num:03d}_{name}"
    batch_dir = batch_jobs_root / dir_name

    batch_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "input").mkdir()
    (batch_dir / "evaluation").mkdir()

    return batch_dir
