# Backend Comparison

This page documents how to compare backends in `llm-batch-pipeline`, both from
local benchmark artifacts and from Grafana.

## Recommended stable comparison target

For `nanu`, the recommended stable live comparison target is:

- Ollama: `qwen3:8b`
- vLLM: `Qwen/Qwen3-8B` served as `qwen3:8b`

Why this pair:

- it fits cleanly on the 3×24 GB RTX 4090 GPUs
- both engines can run as true **3-way sharded one-model-per-GPU** setups
- it avoids the stability problems we hit with `gpt-oss:120b` on vLLM

## Stable 100-sample comparison artifact

Generated artifact:

- `batches/qwen3_8b_100_comparison_backend.json`

Current result:

```json
{
  "model": "qwen3:8b",
  "sample_size": 100,
  "ollama": {
    "requests_per_second": 0.504,
    "row_duration_p50_ms": 5452.196,
    "accuracy": 0.43,
    "macro_f1": 0.3007
  },
  "vllm": {
    "requests_per_second": 0.851,
    "row_duration_p50_ms": 3369.361,
    "accuracy": 0.7,
    "macro_f1": 0.4118
  },
  "label_agreement_rate": 0.73,
  "label_agreement_compared": 100
}
```

Interpretation:

- vLLM is faster on this host for this model and benchmark prompt:
  - throughput: `0.851 req/s` vs `0.504 req/s`
  - p50 row wall-clock: `3369 ms` vs `5452 ms`
- vLLM also scored better on this exact 100-sample benchmark run.
- Cross-backend label agreement is `0.73` on the same 100 inputs.

## Grafana dashboards

### Overview

- `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-overview/llm-batch-pipeline-overview`

Use this to inspect one selected run at a time. It now has a visible
`backend` selector.

### Run Drilldown

- `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-run-drilldown/llm-batch-pipeline-run-drilldown`

Use this to inspect a single run in detail. It now has a visible `backend`
selector and backend-aware run-key selection.

### Run Comparison

- `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-run-comparison/llm-batch-pipeline-run-comparison`

Use this to compare runs within a backend. It now has a visible `backend`
selector and backend-aware hidden `latest_run`, `prev_run_1`, and
`prev_run_2` variables.

### Dedicated Backend Comparison

- `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-backend-comparison/llm-batch-pipeline-backend-comparison`

This dashboard is dedicated to comparing one Ollama run vs one vLLM run.
It exposes two visible selectors:

- `Ollama Run`
- `vLLM Run`

Panels included:

- throughput (`req/s`) per backend
- accuracy per backend
- macro F1 per backend
- row p50 duration per backend
- total pipeline duration per backend
- request p95 latency per backend
- stage duration breakdown for each backend
- warmup duration per shard for each backend
- validation outcomes for each backend
- retry/error log panels for each backend

### Dedicated 3-Way Backend Comparison

- `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-backend-3way/llm-batch-pipeline-backend-comparison-3-way`

This dashboard compares one Ollama run, one vLLM run, and one llama.cpp run.
It exposes three visible selectors:

- `Ollama Run`
- `vLLM Run`
- `llama.cpp Run`

Panels included:

- throughput (`req/s`) per backend
- accuracy per backend
- macro F1 per backend
- row p50 duration per backend
- total pipeline duration per backend
- request p95 latency per backend
- stage duration breakdown for each backend
- warmup duration per shard for each backend
- validation outcomes for each backend
- retry/error log panels for each backend

## How to rerun the benchmark

Use the sequential runbook in:

- `docs/benchmark-run.md`

The helper scripts on `nanu` are described in:

- `scripts/nanu/README.md`

## Experimental note: GPT-OSS

`gpt-oss:120b` / `openai/gpt-oss-120b` is still documented, but it is an
**advanced / experimental** path on `nanu`:

- Ollama can run it via the system `ollama.service`.
- vLLM required a patched startup path and was not stable for the full
  100-sample benchmark.

Use `qwen3:8b` for day-to-day backend speed comparisons.
