# LLM Batch Pipeline

`llm-batch-pipeline` is a generic LLM batch processing pipeline. It discovers and parses input files via a plugin system, renders OpenAI Batch API (or Ollama) requests, validates structured JSON outputs with a Pydantic schema, evaluates predictions against ground truth, and exports results to XLSX/JSON.

See the **Admin Guide** for installation/deployment, and the **Developer Guide** for how to extend the pipeline with custom plugins, prompts, schemas, and evaluation.

## Workflow
```mermaid
flowchart TD
    A[Input Files] --> B[1. Discover]
    B --> C[2. Filter - pre]
    C --> D[3. Transform]
    D --> E[4. Filter - post]
    E --> F[5. Render JSONL]
    F --> G{6. Human Review}
    G -->|Approved| H[7. Submit to Backend]
    G -->|--auto-approve| H
    G -->|Rejected| Z[Abort]
    H --> I[8. Validate Results]
    I --> J[9. Evaluate]
    J --> K[10. Export]

    subgraph Backends
        H --> H1[OpenAI Batch API]
        H --> H2[Ollama Local]
    end

    H1 --> I
    H2 --> I

    subgraph Outputs
        K --> K1[results.xlsx]
        K --> K2[evaluation.xlsx]
        K --> K3[evaluation.json]
        K --> K4[metrics.json]
    end
```

## Admin / Install
- Admin guide: [`docs/admin-guide.md`](docs/admin-guide.md)
- Install (example): `uv sync`

## Requirements
- OpenAI backend: set `OPENAI_API_KEY`.
- Local LLM via Ollama: run an Ollama server (pull the model), then use `--backend ollama --base-url http://HOST:11434` (repeat `--base-url` for multi-server sharding).
- OpenAI API compatible local server (if supported by your server): use `--backend openai` and configure the OpenAI SDK base URL (commonly via `OPENAI_BASE_URL`).

## Test / Benchmark
- SpamAssassin corpus reference run: [`docs/benchmark-run.md`](docs/benchmark-run.md)

## Architecture
- [`docs/architecture.md`](docs/architecture.md)

## Extend (plugins)
- [`docs/developer-guide.md`](docs/developer-guide.md)
- Developer guide covers: custom prompt, custom Pydantic schema, and custom evaluation.

## Monitor
- Prometheus metrics + Grafana: [`docs/admin-guide.md`](docs/admin-guide.md)

## License
- EUPL (v1.2): [see `LICENSE`](LICENSE)
