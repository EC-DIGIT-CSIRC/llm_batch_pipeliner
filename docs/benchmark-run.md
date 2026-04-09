# Benchmark Run: Spam Detection with gemma4:latest

This document records a representative end-to-end benchmark run of the
`llm_batch_pipeline` against a 3-GPU Ollama cluster, using the SpamAssassin
500-email sample corpus.

## Environment

| Component | Detail |
|---|---|
| Pipeline | `llm_batch_pipeline` (local, `new/` directory) |
| Model | `gemma4:latest` |
| Backend | Ollama (3-way sharded) |
| Hardware | `nanu` -- 3x RTX 4090 |
| Ollama instances | `nanu:11435`, `nanu:11436`, `nanu:11437` |
| Python | 3.13 |
| Date | 2026-04-09 |

## Dataset

- **Source**: SpamAssassin public corpus, stratified 500-email sample
- **Location**: `benchmarks/spamassassin/runs/spam_500_sample/input/`
- **Classes**: `ham` (easy\_ham, easy\_ham\_2, hard\_ham) and `spam` (spam, spam\_2)
- **Category map**: filename prefix to label mapping via `category-map.json`

Filenames have hash-like suffixes (e.g.
`easy_ham__00027.4d456dd9ce0afde7629f94dc3034e0bb`), so the spam detection
plugin uses an allowlist/denylist approach in `can_read()` instead of extension
matching.

## Batch Configuration

Batch directory: `batches/batch_002_spam_benchmark/`

```toml
plugin_name = "spam_detection"
model = "gemma4:latest"
backend = "ollama"

base_urls = ["http://nanu:11435", "http://nanu:11436", "http://nanu:11437"]
num_shards = 3
num_parallel_jobs = 3

label_field = "classification"
positive_class = "spam"

auto_approve = true
log_level = "WARN"

request_timeout_seconds = 600
```

The run was invoked with `--log-level DEBUG` on the CLI to get full per-request
diagnostics in the JSONL log file, while the `ConsoleLogFilter` suppressed
per-item events from terminal output.

### Command

```bash
uv run llm-batch-pipeline run \
  --batch-dir batches/batch_002_spam_benchmark \
  --plugin spam_detection \
  --model gemma4:latest \
  --backend ollama \
  --base-url http://nanu:11435 \
  --base-url http://nanu:11436 \
  --base-url http://nanu:11437 \
  --num-parallel-jobs 3 \
  --label-field classification \
  --positive-class spam \
  --auto-approve \
  --log-level DEBUG
```

## Pipeline Stages

| Stage | Duration | Detail |
|---|---|---|
| discover | <1s | 500 files read |
| filter\_1 | <1s | kept 493/500 (7 empty bodies filtered) |
| transform | <1s | 493 files whitespace-trimmed |
| filter\_2 | <1s | kept 493/493 |
| render | <1s | 1 shard, 493 requests |
| review | <1s | auto-approved |
| **submit** | **17m 21s** | **493 ok, 0 failed** |
| validate | <1s | 451 valid, 42 invalid |
| output\_transform | <1s | 451 rows |
| evaluate | <1s | accuracy=0.8581, macro\_f1=0.8573 |
| export | <1s | XLSX files written |

**Total wall time**: ~17m 22s

## Submission Performance

| Metric | Value |
|---|---|
| Total requests | 493 |
| Completed | 493 |
| Failed | 0 |
| Duration | 1020.6s |
| Throughput | 0.48 req/s (~2.06 s/it) |
| Sharding | 3-way interleaved round-robin |

Model warmup took ~7s per server (~21s total, sequential) before the first
batch request.  Interleaved shard submission ensured all 3 GPUs were utilised
concurrently from the start.

## Evaluation Results

### Overall Metrics

| Metric | Value |
|---|---|
| **Accuracy** | **0.8581** |
| **Macro F1** | **0.8573** |
| Macro Precision | 0.8641 |
| Macro Recall | 0.8778 |
| Total Predictions | 451 |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ham | 0.9813 | 0.7778 | 0.8678 | 270 |
| spam | 0.7468 | 0.9779 | 0.8469 | 181 |

### Confusion Matrix

|  | Pred: ham | Pred: spam |
|---|---|---|
| **True: ham** | 210 | 60 |
| **True: spam** | 4 | 177 |

## Analysis

1. **Zero submission failures** -- all 493 requests succeeded.  The previous
   run attempt had 100% timeouts due to bugs (connect timeout too short, no
   model warmup, sequential shard submission).

2. **42 validation errors** (8.5%) -- the model returned output that did not
   parse against the Pydantic schema.  These are excluded from evaluation.
   Potential causes: malformed JSON, missing required fields, or enum values
   outside the allowed set.  Worth investigating whether prompt engineering or
   schema relaxation can reduce this rate.

3. **Spam-aggressive classification** -- the model has very high spam recall
   (97.8%) but classifies 22.2% of legitimate emails as spam (60/270 ham
   misclassified).  This is consistent with a conservative "when in doubt, call
   it spam" strategy.

4. **Excellent spam catch rate** -- only 4 spam emails out of 181 were missed.

5. **Throughput** -- 2.06 s/it across 3x RTX 4090 with `gemma4:latest` is
   reasonable.  The interleaved round-robin submission keeps all GPUs busy.

## Output Artifacts

All outputs are stored under `batches/batch_002_spam_benchmark/`:

```
batches/batch_002_spam_benchmark/
  config.toml            # Batch configuration
  input/                 # 500 symlinks to SpamAssassin emails
  job/
    batch-00001.jsonl    # Rendered JSONL shard (493 requests)
    batch.jsonl          # Symlink to single shard
  output/
    batch-00001-output.jsonl  # Raw Ollama responses
  results/
    validated.json       # 451 validated result rows
  evaluation/
    category-map.json    # Ground truth prefix mapping
  export/
    results.xlsx         # All validated results
    evaluation.xlsx      # Evaluation metrics + confusion matrix
    evaluation.json      # Machine-readable evaluation
  logs/
    pipeline.jsonl       # Full structured log (all per-item events)
```

## Bugs Fixed Before This Run

This was the first successful run.  The following bugs were identified and
fixed in the preceding session:

1. **Connect timeout too short (30s)** -- `httpx.Timeout` was using a separate
   30s connect timeout that expired before Ollama loaded the model.  Fixed by
   using a single uniform timeout matching `request_timeout_seconds` (600s).

2. **No model warmup** -- first batch of requests all failed while the model
   loaded.  Fixed by adding `_warmup_server()` that sends a trivial chat
   request to each server before batch dispatch.

3. **httpx.Client created per retry** -- prevented connection reuse.  Fixed by
   creating the client once outside the retry loop.

4. **Shard submissions not interleaved** -- all futures for shard 0 submitted
   first (FIFO), so GPUs ran sequentially.  Fixed by interleaving submissions
   round-robin: `shard0[0], shard1[0], shard2[0], shard0[1], ...`.

5. **Console spam from per-item log events** -- every request printed to
   terminal.  Fixed by adding `ConsoleLogFilter` that suppresses INFO events
   carrying `file_id` or `custom_id` from the console handler while preserving
   them in JSONL.

6. **Missing per-request URL and success logging** -- couldn't diagnose which
   shard was failing.  Fixed by adding `url=` to all log events and logging
   successes at DEBUG level.
