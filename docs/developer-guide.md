# LLM Batch Pipeline — Developer Guide

Guide for extending `llm-batch-pipeline` with custom plugins, backends, and stages.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                      │
│  argparse → config builder → stage dispatch              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Pipeline Orchestrator (pipeline.py)         │
│  Stage runner, context, retry, resume, dry-run           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               Stage Wiring (stages.py)                   │
│  10 stage functions, each receives PipelineContext        │
└──┬───────┬───────┬───────┬───────┬───────┬──────────────┘
   │       │       │       │       │       │
   ▼       ▼       ▼       ▼       ▼       ▼
Plugins  Filters  Transforms  Render  Backends  Evaluation
```

### Key Design Decisions

1. **Plugin system** — Python ABCs (no metaclass magic, no decorators, no framework). Users subclass `FileReader`, `Filter`, `Transformer`, `OutputTransformer`.

2. **Pipeline orchestrator** — Custom lightweight runner (~200 lines). No framework dependencies. Supports retry, resume from stage, dry-run.

3. **Stage communication** — Stages share state via `PipelineContext.artifacts` (dict) and `PipelineContext.files` / `PipelineContext.filtered_files` (lists of `ParsedFile`).

4. **Schema convention** — Schema files must define `class mySchema(BaseModel)`. Loaded dynamically via `importlib.util.spec_from_file_location`.

5. **Backend abstraction** — `BatchBackend` ABC with `submit()` method. OpenAI and Ollama implementations share common validation/IO utilities in `backends/common.py`.

## Creating a Custom Plugin

### Step 1: Define a FileReader

```python
from pathlib import Path
from llm_batch_pipeline.plugins.base import FileReader, ParsedFile

class CsvReader(FileReader):
    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def read(self, path: Path) -> ParsedFile:
        content = path.read_text(encoding="utf-8")
        return ParsedFile(
            filename=path.name,
            raw_path=path,
            content=content,
            metadata={"rows": content.count("\n")},
        )

    def package_for_llm(self, parsed: ParsedFile) -> str:
        # Format content for the LLM prompt
        return f"CSV Data:\n{parsed.content}"
```

### Step 2: Define Filters (Optional)

```python
from llm_batch_pipeline.plugins.base import Filter, ParsedFile

class MinRowFilter(Filter):
    def __init__(self, min_rows: int = 2):
        self.min_rows = min_rows

    def apply(self, parsed: ParsedFile) -> tuple[bool, str]:
        rows = parsed.metadata.get("rows", 0)
        if rows < self.min_rows:
            return False, f"Too few rows ({rows} < {self.min_rows})"
        return True, "passed"
```

### Step 3: Define Transformers (Optional)

```python
from llm_batch_pipeline.plugins.base import Transformer, ParsedFile

class NormalizeHeadersTransformer(Transformer):
    def apply(self, parsed: ParsedFile) -> ParsedFile:
        lines = parsed.content.split("\n")
        if lines:
            lines[0] = lines[0].lower()
            parsed.content = "\n".join(lines)
        return parsed
```

### Step 4: Define a Pydantic Schema

```python
# my_plugin/schema.py
from pydantic import BaseModel, Field
from typing import Literal

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the text."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score."
    )
    reason: str = Field(description="Explanation.")

# REQUIRED: the pipeline loads schemas by this name
mySchema = SentimentResult
```

### Step 5: Register the Plugin

```python
# my_plugin/plugin.py
from llm_batch_pipeline.plugins.registry import PluginSpec, register_plugin

def register():
    register_plugin(PluginSpec(
        name="csv_sentiment",
        reader=CsvReader(),
        pre_filters=[MinRowFilter(min_rows=2)],
        transformers=[NormalizeHeadersTransformer()],
        post_filters=[],
        default_schema_module="my_plugin.schema",
    ))
```

### Step 6: Add to Auto-Discovery

Add your module path to `_BUILTIN_MODULES` in `src/llm_batch_pipeline/plugins/registry.py`:

```python
_BUILTIN_MODULES = (
    "llm_batch_pipeline.examples.spam_detection.plugin",
    "llm_batch_pipeline.examples.gdpr_detection.plugin",
    "my_plugin.plugin",  # Add your plugin
)
```

Or register programmatically before calling the pipeline:

```python
from my_plugin.plugin import register
register()
```

## Plugin ABCs Reference

### `FileReader`

| Method | Signature | Description |
|--------|-----------|-------------|
| `can_read` | `(path: Path) -> bool` | Return True if this reader handles the file |
| `read` | `(path: Path) -> ParsedFile` | Parse file into a `ParsedFile` |
| `package_for_llm` | `(parsed: ParsedFile) -> str` | Serialize content for the LLM user message |

### `Filter`

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | property `-> str` | Human-readable name (default: class name) |
| `apply` | `(parsed: ParsedFile) -> tuple[bool, str]` | Return `(keep, reason)` |

### `Transformer`

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | property `-> str` | Human-readable name (default: class name) |
| `apply` | `(parsed: ParsedFile) -> ParsedFile` | Transform and return (possibly new) ParsedFile |

### `OutputTransformer`

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | property `-> str` | Human-readable name |
| `apply` | `(rows: list[dict]) -> list[dict]` | Transform validated LLM output rows |

## Stage Contracts

Each stage function has the signature: `(ctx: PipelineContext) -> StageResult`

### Data Flow Between Stages

| Stage | Reads From | Writes To |
|-------|-----------|-----------|
| `discover` | `ctx.config.input_dir` | `ctx.files` |
| `filter_1` | `ctx.files` | `ctx.filtered_files` |
| `transform` | `ctx.filtered_files` | `ctx.filtered_files` (mutated) |
| `filter_2` | `ctx.filtered_files` | `ctx.filtered_files` |
| `render` | `ctx.filtered_files` | `ctx.artifacts["shard_paths"]` |
| `human_review` | `ctx.artifacts` | (blocks until approved) |
| `submit` | `ctx.artifacts["shard_paths"]` | `ctx.artifacts["output_files"]` |
| `validate` | `ctx.artifacts["output_files"]` | `ctx.artifacts["validated_rows"]` |
| `evaluate` | `ctx.artifacts["validated_rows"]` | `ctx.artifacts["eval_report"]` |
| `export` | `ctx.artifacts["validated_rows"]`, `ctx.artifacts["eval_report"]` | XLSX files |

### PipelineContext

```python
@dataclass
class PipelineContext:
    batch_dir: Path
    config: BatchConfig
    console: Console
    metrics: MetricsCollector
    files: list[ParsedFile]           # Set by discover
    filtered_files: list[ParsedFile]  # Set by filter stages
    artifacts: dict[str, Any]         # Shared between stages
```

## Testing

### Running Tests

```bash
uv run pytest              # All tests
uv run pytest -v           # Verbose
uv run pytest -k "spam"    # Filter by name
uv run pytest --tb=short   # Short tracebacks
```

### Writing Tests

Tests live in `tests/` with one test file per source module. Use fixtures from `tests/conftest.py` for sample `.eml` data.

```python
# tests/test_my_plugin.py
from pathlib import Path
from my_plugin.plugin import CsvReader, MinRowFilter

class TestCsvReader:
    def test_can_read_csv(self):
        reader = CsvReader()
        assert reader.can_read(Path("test.csv"))
        assert not reader.can_read(Path("test.txt"))

    def test_read_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        reader = CsvReader()
        parsed = reader.read(csv_file)
        assert parsed.filename == "data.csv"
        assert parsed.metadata["rows"] == 3
```

### Test Conventions

- Prefer fixture-driven tests with real sample data
- Each test file mirrors the source file it tests
- Use `tmp_path` for filesystem tests
- Use `pytest.approx()` for floating-point comparisons

## Linting

### Ruff

```bash
uv run ruff check src/ tests/       # Check
uv run ruff check --fix src/ tests/  # Auto-fix
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py313"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]
```

### Pylint

```bash
uv run pylint src/llm_batch_pipeline/
```

Configuration in `.pylintrc`. Target: 10.00/10.

## Code Conventions

### Import Order

Enforced by ruff `I001`:
1. `from __future__ import annotations`
2. Standard library
3. Third-party
4. Local package

### Naming

- Module-level loggers: `logger = logging.getLogger("llm_batch_pipeline.<module>")`
- Schema convention: `mySchema = MyActualModelClass`
- Short variable names permitted: `cm`, `wb`, `ws`, `fh`, `pf`, `h`, `n`, `d`, `t` (see `.pylintrc` `good-names`)

### Error Handling

- Stage functions catch exceptions and return `StageResult(status="failed", error=str(e))`
- Broad `except Exception` is acceptable in plugin auto-discovery and per-request processing (resilience)
- Always log drop reasons as structured metadata

### Atomic File Writes

Use `write_json_atomic()` and `write_text_atomic()` from `backends/common.py` for output files to prevent partial writes.

## CI Workflows

Three GitHub Actions workflows in `.github/workflows/`:

| Workflow | Tool | Purpose |
|----------|------|---------|
| `ruff.yml` | Ruff | Import sorting, style, modern Python |
| `pylint.yml` | Pylint | Code quality, naming, complexity |
| `semgrep.yml` | Semgrep | Security patterns, SAST |

All three must pass before merging.
