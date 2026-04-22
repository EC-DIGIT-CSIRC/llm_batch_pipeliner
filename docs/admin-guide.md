# LLM Batch Pipeline — Admin Guide

Operations and deployment guide for running `llm-batch-pipeline` in production.

## Prerequisites

- Python 3.13+
- `uv` package manager
- For OpenAI backend: `OPENAI_API_KEY` environment variable
- For Ollama backend: One or more Ollama servers with models pulled

## Installation

```bash
uv sync              # Install all dependencies
uv sync --group dev  # Include dev tools (pytest, ruff, pylint)
```

### Dependency Verification

All dependencies are pinned in `uv.lock`. The project follows a strict supply chain policy:

- Versions and hashes are pinned
- Packages published < 48 hours ago are flagged
- Maintainer account changes are flagged
- Typosquatting names are checked against popular packages
- GPL/AGPL dependencies are permitted

### Production Dependencies

| Package | Purpose | License |
|---------|---------|---------|
| `httpx` | Ollama HTTP client | BSD-3 |
| `openai` | OpenAI API client | Apache-2.0 |
| `opentelemetry-sdk` | Metrics and log SDK | Apache-2.0 |
| `opentelemetry-exporter-otlp-proto-http` | OTLP/HTTP exporter | Apache-2.0 |
| `openpyxl` | XLSX export | MIT |
| `pydantic` | Schema validation | MIT |
| `python-dotenv` | Environment file loading | BSD-3 |
| `rich` | Terminal UI | MIT |
| `selectolax` | HTML parsing | MIT |
| `charset-normalizer` | Charset fallback | MIT |
| `mail-parser-reply` | Reply chain stripping | Apache-2.0 |

## Deployment Patterns

### Single Machine with Ollama

```bash
# Pull model
ollama pull llama3.1:8b

# Run pipeline
uv run llm-batch-pipeline run \
    --batch-dir batches/batch_001_test \
    --plugin spam_detection \
    --backend ollama \
    --model llama3.1:8b \
    --num-parallel-jobs 4 \
    --auto-approve
```

### Multi-GPU Server Farm

For servers with multiple GPUs (each running an Ollama instance):

```bash
uv run llm-batch-pipeline run \
    --batch-dir batches/batch_001_test \
    --plugin spam_detection \
    --backend ollama \
    --base-url http://gpu1:11434 \
    --base-url http://gpu2:11434 \
    --base-url http://gpu3:11434 \
    --num-parallel-jobs 4 \
    --model llama3.1:70b \
    --auto-approve
```

The pipeline automatically:
1. Shards the JSONL across servers (round-robin)
2. Creates per-shard thread pools
3. Aggregates results back into a single output

### OpenAI Batch API (Cloud)

```bash
export OPENAI_API_KEY="sk-..."

uv run llm-batch-pipeline run \
    --batch-dir batches/batch_001_test \
    --plugin spam_detection \
    --backend openai \
    --model gpt-4o-mini \
    --poll-interval 30 \
    --auto-approve
```

OpenAI batches have a 24h completion window. Use `--no-wait` to submit and check later:

```bash
# Submit without waiting
uv run llm-batch-pipeline submit \
    --batch-dir batches/batch_001_test \
    --backend openai \
    --no-wait

# Resume monitoring later
uv run llm-batch-pipeline submit \
    --batch-dir batches/batch_001_test \
    --backend openai \
    --resume-batch-id batch_abc123
```

## Monitoring

### OpenTelemetry Collector + Prometheus + Grafana

The pipeline sends OTLP telemetry to an OpenTelemetry Collector.

Set these environment variables before running the pipeline:

```bash
export OTEL_SERVICE_NAME=llm-batch-pipeline
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

Concrete shared test setup:

```bash
export OTEL_SERVICE_NAME=llm-batch-pipeline
export OTEL_EXPORTER_OTLP_ENDPOINT=http://bee.lo-res.org:4318
```

The Collector currently deployed on `bee.lo-res.org` exposes:
- OTLP/HTTP ingest: `http://bee.lo-res.org:4318`
- Prometheus scrape endpoint: `http://bee.lo-res.org:9464/metrics`
- Traefik-routed Prometheus API: `https://bee.lo-res.org/prometheus`
- Traefik-routed Loki API: `https://bee.lo-res.org/loki`

The deployed Collector config is tracked in this repository at `docs/otel-collector-bee.yaml`.
The Bee-side Prometheus/Loki Docker Compose file is tracked at `docs/bee-observability-compose.yml`.

Grafana test instance:
- URL: `http://grafana-test.intelmq.org:3000`
- Datasources created: `Prometheus Bee`, `Loki Bee`
- Dashboard imported: `LLM Batch Pipeline Overview`
- Dashboard URL: `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-overview/llm-batch-pipeline-overview`
- Comparison dashboard imported: `LLM Batch Pipeline Run Comparison`
- Comparison dashboard URL: `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-run-comparison/llm-batch-pipeline-run-comparison`
- Drilldown dashboard imported: `LLM Batch Pipeline Run Drilldown`
- Drilldown dashboard URL: `http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-run-drilldown/llm-batch-pipeline-run-drilldown`
- Dashboard definition in repo: `docs/llm-batch-pipeline-dashboard.json`
- Comparison dashboard definition in repo: `docs/llm-batch-pipeline-run-comparison-dashboard.json`
- Drilldown dashboard definition in repo: `docs/llm-batch-pipeline-run-drilldown-dashboard.json`

Dashboard notes:
- The detailed per-stage and per-row panels require pipeline runs with `--log-level INFO`.
- The dashboard has a `Run Key` textbox for the selected-run panels.
- The default `Run Key` is seeded to the latest verified run when the dashboard JSON is imported.
- To inspect an older run, copy its `service_run_key` from the `Recent Completed Runs` panel and paste it into the `Run Key` textbox.

Relevant upstream docs:
- OpenTelemetry Collector: <https://opentelemetry.io/docs/collector/>
- Prometheus with OpenTelemetry: <https://prometheus.io/docs/guides/opentelemetry/>

TL;DR local setup:

1. Start an OpenTelemetry Collector that receives OTLP over HTTP and exposes Prometheus metrics:

   ```yaml
   receivers:
     otlp:
       protocols:
         http:
           endpoint: 0.0.0.0:4318

   processors:
     batch:

   exporters:
     prometheus:
       endpoint: 0.0.0.0:9464
     debug:
       verbosity: normal

   service:
     pipelines:
       metrics:
         receivers: [otlp]
         processors: [batch]
         exporters: [prometheus]
       logs:
         receivers: [otlp]
         processors: [batch]
         exporters: [debug]
   ```

2. Run the Collector:

   ```bash
   docker run --rm \
     -p 4318:4318 \
     -p 9464:9464 \
     -v "$(pwd)/otel-collector.yaml:/etc/otelcol-contrib/config.yaml" \
     otel/opentelemetry-collector-contrib
   ```

3. Point Prometheus at the Collector's Prometheus exporter:

   ```yaml
   scrape_configs:
     - job_name: llm-batch-pipeline
       static_configs:
         - targets: ['localhost:9464']
   ```

   Replace `localhost` with the Collector hostname if Prometheus runs on a different machine or container network.

4. Add that Prometheus server as a Grafana datasource.

To verify the pipeline is exporting telemetry to the shared `bee` collector:

```bash
curl -fsS http://bee.lo-res.org:9464/metrics | rg "llm_batch_pipeline"
ssh bee "docker logs --since 5m otel-collector-bee 2>&1" | rg "llm_batch_pipeline|llm-batch-pipeline"
```

Prometheus-visible metric names:

| Metric | Type | Description |
|--------|------|-------------|
| `llm_batch_pipeline_runs_total` | Counter | Full pipeline runs by status |
| `llm_batch_pipeline_stage_duration_seconds` | Histogram | Duration per pipeline stage |
| `llm_batch_pipeline_requests_total` | Counter | LLM requests by backend/model/status |
| `llm_batch_pipeline_request_duration_seconds` | Histogram | LLM request latency |
| `llm_batch_pipeline_active_requests` | UpDownCounter | In-flight LLM requests |
| `llm_batch_pipeline_validation_total` | Counter | Validation rows by status |

Prometheus stores metrics only. If you want logs in Grafana, add a log backend such as Loki to the Collector or use Grafana Alloy and route the OTLP log stream there.

For the shared Bee setup, Loki is already wired behind the Collector and routed through Traefik. The imported Grafana dashboard uses Loki for per-run and per-stage analytics because batch-job logs preserve the event timestamps needed for success-rate and average-duration queries over a selected time range.

### Log Files

All runs produce structured JSONL logs in the batch's `logs/` directory:

```
logs/
├── pipeline.jsonl    # Structured event log (every step)
└── metrics.json      # Aggregated timing and count metrics
```

Each log line contains:

```json
{
  "timestamp": "2026-04-08T12:34:56.789Z",
  "level": "info",
  "logger": "llm_batch_pipeline.stages",
  "step": "discover",
  "status": "ok",
  "duration_ms": 42.5,
  "message": "Discovered 500 files"
}
```

## Security

### TLS

By default, all HTTPS connections verify TLS certificates. Use `--insecure` / `-k` only for development:

```bash
# Development only — disables TLS verification
uv run llm-batch-pipeline submit --insecure ...
```

### API Keys

Store API keys in `.env` files (loaded via `python-dotenv`) or environment variables. Never commit `.env` files.

### File Permissions

Output directories are created with default permissions. On shared systems, consider restricting access:

```bash
chmod 700 batches/batch_001_sensitive/
```

## Batch Directory Lifecycle

```
1. init         → Creates batch_NNN_name/ with input/, evaluation/, config.toml
2. populate     → User copies input files into input/
3. run/render   → Creates job/ with JSONL shards
4. submit       → Creates output/ with LLM responses
5. validate     → Creates results/ with validated JSON
6. evaluate     → Creates export/evaluation.json
7. export       → Creates export/*.xlsx
8. archive      → User archives or deletes the batch directory
```

### Cleaning Up

Batch directories are self-contained. To remove a completed batch:

```bash
rm -rf batches/batch_001_test/
```

## Troubleshooting

### Common Issues

**"No plugins registered"** — Ensure you installed the package (`uv sync`). Built-in plugins (`spam_detection`, `gdpr_detection`) are auto-registered on import.

**"Input directory not found"** — The `input/` subdirectory must exist in the batch directory and contain files the plugin's reader can handle.

**"Batch directory not found"** — The `--batch-dir` argument is resolved relative to `--batch-jobs-root` (default: `./batches`). You can also pass an absolute or relative path directly.

**OpenAI rate limits** — The Batch API has its own rate limits separate from the real-time API. Check your OpenAI dashboard for quota.

**Ollama timeouts** — Increase `--request-timeout` for large models or slow hardware. The default is 600 seconds.

**Schema validation failures** — Ensure the LLM model supports structured output. Check `results/validated.json` for per-row error details.
