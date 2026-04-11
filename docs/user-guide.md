# LLM Batch Pipeline — User Guide

Generic LLM batch processing pipeline with pluggable preprocessing, validation, evaluation, and export stages.

## Overview

`llm-batch-pipeline` takes input files (emails, documents, etc.), preprocesses them through a configurable plugin system, renders OpenAI Batch API or Ollama request payloads, submits them to an LLM backend, validates the results against a Pydantic schema, evaluates accuracy against ground truth, and exports everything to XLSX and JSON.

### Pipeline Stages

```
1. Discover     → Scan input directory, parse files via plugin reader
2. Filter (pre) → Apply plugin filter chain, log drops
3. Transform    → Apply plugin transform chain
4. Filter (post)→ Optional second filter pass
5. Render       → Build OpenAI Batch API JSONL with sharding
6. Human Review → Show stats, wait for confirmation (--auto-approve to skip)
7. Submit       → Send to OpenAI Batch API or Ollama
8. Validate     → Schema-validate each LLM response (Pydantic)
9. Evaluate     → Confusion matrix, precision/recall/F1, ROC
10. Export      → XLSX workbooks + JSON metrics
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd new/

# Install with uv
uv sync

# Verify
uv run llm-batch-pipeline --version
```

## Quick Start

### 1. Create a Batch Job

```bash
uv run llm-batch-pipeline init my_spam_test \
    --plugin spam_detection \
    --model gpt-4o-mini
```

This creates `batches/batch_001_my_spam_test/` with:

```
batch_001_my_spam_test/
├── config.toml      # Batch configuration
├── input/           # Place input files here
└── evaluation/      # Place ground-truth files here
```

### 2. Add Input Files

Copy your `.eml` files into the `input/` directory:

```bash
cp /path/to/emails/*.eml batches/batch_001_my_spam_test/input/
```

### 3. Run the Full Pipeline

```bash
# With OpenAI backend
uv run llm-batch-pipeline run \
    --batch-dir batch_001_my_spam_test \
    --plugin spam_detection \
    --model gpt-4o-mini \
    --auto-approve

# With Ollama backend (local)
uv run llm-batch-pipeline run \
    --batch-dir batch_001_my_spam_test \
    --plugin spam_detection \
    --backend ollama \
    --model llama3.1:8b \
    --base-url http://localhost:11434 \
    --auto-approve
```

### 4. Run Individual Stages

Instead of running the full pipeline, you can run stages individually:

```bash
# Render JSONL only (discover → filter → transform → render)
uv run llm-batch-pipeline render \
    --batch-dir batch_001_my_spam_test \
    --plugin spam_detection

# Submit to backend
uv run llm-batch-pipeline submit \
    --batch-dir batch_001_my_spam_test \
    --backend ollama \
    --model llama3.1:8b

# Validate results
uv run llm-batch-pipeline validate \
    --batch-dir batch_001_my_spam_test \
    --schema-file path/to/schema.py

# Evaluate against ground truth
uv run llm-batch-pipeline evaluate \
    --batch-dir batch_001_my_spam_test \
    --category-map path/to/categories.json

# Export to XLSX
uv run llm-batch-pipeline export \
    --batch-dir batch_001_my_spam_test
```

## CLI Reference

### Global Options

| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `--batch-dir DIR` | Batch job directory (auto-resolved by name) |
| `--batch-jobs-root DIR` | Root for batch dirs (default: `./batches`) |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

### `init` — Create a Batch Job

```bash
uv run llm-batch-pipeline init <name> --plugin <plugin> [--model <model>] [--prompt-file <path>] [--schema-file <path>]
```

Batch directories are auto-numbered: `batch_001_<name>`, `batch_002_<name>`, etc.

### `run` — Full Pipeline

```bash
uv run llm-batch-pipeline run --batch-dir <dir> --plugin <plugin> [options]
```

| Flag | Description |
|------|-------------|
| `--plugin NAME` | Plugin to use (required) |
| `--model NAME` | LLM model (default: `gpt-4o-mini`) |
| `--backend TYPE` | `openai` or `ollama` (default: `openai`) |
| `--auto-approve` | Skip human review stage |
| `--start-from STAGE` | Resume from a specific stage |
| `--dry-run` | Show plan without executing |
| `--input-dir DIR` | Override input directory |

### `submit` — Submit to Backend

| Flag | Description |
|------|-------------|
| `--backend TYPE` | `openai` or `ollama` |
| `--base-url URL` | Ollama server URL (repeatable for sharding) |
| `--num-shards N` | Number of Ollama shards |
| `--num-parallel-jobs N` | Parallel jobs per shard (default: 3) |
| `--request-timeout N` | Per-request timeout in seconds (default: 600) |
| `--poll-interval N` | OpenAI batch poll interval (default: 15s) |
| `--batch-jsonl PATH` | Explicit JSONL file to submit |
| `--resume-batch-id ID` | Resume monitoring an existing OpenAI batch |
| `--prompt-override TEXT` | Override prompt at submit time |
| `--prompt-override-file PATH` | File with prompt override |
| `-k` / `--insecure` | Disable TLS verification |
| `--no-wait` | Submit without waiting for completion |

### `evaluate` — Evaluation

| Flag | Description |
|------|-------------|
| `--ground-truth-csv PATH` | CSV with `(filename, label)` columns |
| `--category-map PATH` | JSON mapping filename prefixes to labels |
| `--label-field NAME` | Schema field containing the predicted label |
| `--confidence-field NAME` | Schema field containing the confidence score |
| `--positive-class NAME` | Positive class for binary ROC/AUC |

### `list` — Show Plugins

```bash
uv run llm-batch-pipeline list
```

## Configuration

### config.toml

Each batch directory can contain a `config.toml` that sets defaults:

```toml
plugin_name = "spam_detection"
model = "gpt-4o-mini"
prompt_file = "prompt.txt"
schema_file = "schema.py"
backend = "ollama"
base_urls = ["http://gpu1:11434", "http://gpu2:11434"]
num_parallel_jobs = 4
auto_approve = true
```

CLI arguments always override TOML values.

### Pydantic Schema Files

Schema files must define a class named `mySchema` inheriting from `pydantic.BaseModel`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SpamDetectionResult(BaseModel):
    classification: Literal["spam", "ham"] = Field(
        description="Whether the email is spam or legitimate."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    reason: str = Field(description="Explanation for the classification.")

# This alias is required — the pipeline loads schemas by this name
mySchema = SpamDetectionResult
```

The pipeline automatically:
- Converts the schema to JSON Schema with `additionalProperties: false` and all properties required (strict mode)
- Sanitizes the schema for Ollama's GBNF grammar (strips `minimum`, `maximum`, `minItems`, etc.)
- Infers `label_field` and `confidence_field` from the schema if not explicitly set

### Prompt Files

Plain text files used as the system prompt / instructions:

```text
You are an email security analyst. Classify the following email as either
"spam" or "ham" (legitimate). Provide your confidence level and reasoning.
```

### Ground Truth

Two methods for providing ground truth labels:

**CSV file** (`--ground-truth-csv`):
```csv
filename,label
email_001.eml,spam
email_002.eml,ham
```

**Category map** (`--category-map`): JSON mapping filename prefixes to labels:
```json
{
    "spam_": "spam",
    "ham_": "ham",
    "phish_": "phishing"
}
```

## Backends

### OpenAI Batch API

The default backend. Submits to OpenAI's `/v1/responses` endpoint with file upload, polls for completion, and downloads results.

```bash
export OPENAI_API_KEY="sk-..."
uv run llm-batch-pipeline submit --backend openai --batch-dir my_batch
```

### Ollama (Local)

For local inference using Ollama servers. Supports multi-server round-robin sharding with per-shard thread pools.

```bash
# Single server
uv run llm-batch-pipeline submit \
    --backend ollama \
    --base-url http://localhost:11434 \
    --model llama3.1:8b

# Multi-server sharding (3 GPUs)
uv run llm-batch-pipeline submit \
    --backend ollama \
    --base-url http://gpu1:11434 \
    --base-url http://gpu2:11434 \
    --base-url http://gpu3:11434 \
    --num-parallel-jobs 4 \
    --model llama3.1:70b
```

## Output Structure

After a full pipeline run, the batch directory contains:

```
batch_001_my_test/
├── config.toml
├── input/              # Original input files
├── evaluation/         # Ground truth files
├── job/                # Rendered JSONL shards
│   ├── batch-00001.jsonl
│   └── batch.jsonl → batch-00001.jsonl
├── output/             # Raw LLM responses
│   ├── output.jsonl
│   └── summary.json
├── results/            # Validated results
│   └── validated.json
├── export/             # XLSX and JSON exports
│   ├── results.xlsx
│   ├── evaluation.xlsx
│   └── evaluation.json
└── logs/               # Structured logs
    ├── pipeline.jsonl
    └── metrics.json
```

## Built-in Plugins

### `spam_detection`

Classifies emails as spam or ham.

- **Reader**: Parses `.eml`, `.txt`, and `.msg` files; extracts headers, plain text, and HTML (via selectolax)
- **Filters**: `EmptyBodyFilter` — drops emails with no body text
- **Transforms**: `TrimWhitespaceTransformer` — collapses excessive whitespace
- **Schema**: `SpamDetectionResult` — classification, confidence, reason, indicators, suspicious URLs, sender analysis

### `gdpr_detection`

Detects personally identifiable information (PII) in emails.

- **Reader**: Extends `EmailReader` with attachment filename extraction
- **Filters**: `MinLengthFilter` (drops emails < 20 chars), `AutoReplyFilter` (drops auto-replies)
- **Transforms**: `RedactAttachmentNamesTransformer` — replaces filenames with placeholders
- **Schema**: `GdprDetectionResult` — contains_pii, sensitivity_level, confidence, PII categories, recommended action

## Metrics and Observability

### OTLP Metrics and Logs

Telemetry is exported to an OpenTelemetry Collector when standard OTLP environment variables are set:

```bash
export OTEL_SERVICE_NAME=llm-batch-pipeline
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

The pipeline emits these metric series:
- `llm_batch_pipeline_runs_total`
- `llm_batch_pipeline_stage_duration_seconds`
- `llm_batch_pipeline_requests_total`
- `llm_batch_pipeline_request_duration_seconds`
- `llm_batch_pipeline_active_requests`
- `llm_batch_pipeline_validation_total`

Structured logs continue to be written locally to `logs/pipeline.jsonl` and are also sent to the Collector when OTLP log export is configured.

### Local Metrics

Even without the HTTP server, a `metrics.json` summary is always written to the logs directory after each run.
