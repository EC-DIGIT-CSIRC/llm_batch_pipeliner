# Running vLLM

This guide covers running `llm-batch-pipeline` against [vLLM](https://docs.vllm.ai)
servers, including:

- 3-way sharded one-model-per-GPU setups
- one-endpoint tensor/pipeline-parallel setups for very large models

## Why vLLM

- Native OpenAI-compatible HTTP server (`/v1/responses`, `/v1/chat/completions`).
- High-throughput continuous batching, paged attention.
- First-class JSON Schema structured output (`response_format=json_schema`).
- Built-in support for tensor-parallel and data-parallel sharding.

## Backend choice

Use `--backend vllm`. The vLLM backend mirrors the Ollama backend's surface
(`--base-url` repeatable for sharding, identical telemetry, identical retry/
warmup semantics) so you can swap between them with no other CLI changes.

## Endpoint selection

The backend defaults to `/v1/responses`, which means the pipeline's rendered
JSONL needs **no translation**. Override with `--vllm-endpoint chat` to target
`/v1/chat/completions` instead (translation is done client-side).

## Authentication

vLLM can be started with `--api-key <token>`. The pipeline reads the token
from either:

1. `--api-key <token>` on the CLI, or
2. the `VLLM_API_KEY` environment variable.

If neither is set, no `Authorization` header is sent.

## Single server

```bash
# On the GPU host:
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-2-9b-it \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
    --served-model-name gemma4:latest

# On your client:
uv run llm-batch-pipeline run \
    --batch-dir <batch> \
    --backend vllm \
    --base-url http://gpu-host:8000 \
    --model gemma4:latest \
    --auto-approve
```

## 3-way sharded on a single host (one model per GPU)

Use `scripts/nanu/vllm-shards-up.sh` (also see [benchmark-run.md](benchmark-run.md)):

```bash
ssh nanu '/home/aaron/nanu-scripts/vllm-shards-up.sh google/gemma-2-9b-it gemma4:latest'
```

This starts three independent `vllm serve` processes pinned to GPUs 0/1/2,
listening on ports 18000/18001/18002. Each is invoked as:

```bash
CUDA_VISIBLE_DEVICES=N vllm serve <weights> \
    --host 0.0.0.0 --port 1800N \
    --tensor-parallel-size 1 \
    --served-model-name gemma4:latest \
    --gpu-memory-utilization 0.9 \
    --dtype auto
```

Then run the pipeline:

```bash
uv run llm-batch-pipeline run \
    --batch-dir <batch> \
    --backend vllm \
    --base-url http://nanu:18000 \
    --base-url http://nanu:18001 \
    --base-url http://nanu:18002 \
    --num-shards 3 \
    --num-parallel-jobs 1 \
    --model gemma4:latest \
    --auto-approve
```

Stop the shards when done:

```bash
ssh nanu /home/aaron/nanu-scripts/vllm-shards-down.sh
```

## Tensor parallel for one large model

If a model does not fit on a single GPU, run **one** vLLM server with
`--tensor-parallel-size N`. From the pipeline's perspective this is just one
`--base-url`. No sharding flags needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve big-model \
    --tensor-parallel-size 4 --host 0.0.0.0 --port 8000

uv run llm-batch-pipeline run --backend vllm --base-url http://gpu-host:8000 ...
```

## Tensor-parallel no-sharding benchmark: `qwen3.6:latest`

For a heavier single-endpoint benchmark on `nanu`, use the AWQ quantized HF
weights and serve them under the same logical model name as Ollama.

Recommended pair:

- Ollama: `qwen3.6:latest`
- vLLM: `QuantTrio/Qwen3.6-35B-A3B-AWQ`, served as `qwen3.6:latest`

Suggested vLLM launch (GPUs 0 and 1 only):

```bash
ssh nanu 'VLLM_PYTHON=$HOME/git/vllm/.venv/bin/python VLLM_VISIBLE_DEVICES=0,1 VLLM_PORT=18040 VLLM_TP=2 VLLM_PP=1 VLLM_MAX_MODEL_LEN=32768 VLLM_GPU_MEMORY_UTILIZATION=0.9 VLLM_EXTRA_ARGS="--enable-expert-parallel --reasoning-parser qwen3 --language-model-only" /home/aaron/nanu-scripts/vllm-tp-up.sh QuantTrio/Qwen3.6-35B-A3B-AWQ qwen3.6:latest'

uv run llm-batch-pipeline run \
    --batch-dir <batch> \
    --backend vllm \
    --base-url http://nanu:18040 \
    --num-shards 1 \
    --num-parallel-jobs 1 \
    --model qwen3.6:latest \
    --plugin spam_detection \
    --request-timeout 1200 \
    --auto-approve

ssh nanu 'VLLM_PORT=18040 /home/aaron/nanu-scripts/vllm-tp-down.sh'
```

This is the no-sharding / tensor-parallel path referenced from
`docs/benchmark-run.md`.

## Large MoE models that need pipeline parallelism

Some large MoE models cannot be split cleanly across 3 GPUs with tensor
parallelism alone. `gpt-oss-120b` is the main example in this repo.

- `num_attention_heads = 64`, so tensor parallel size must divide 64.
- `tensor-parallel-size=3` is invalid.
- On 3×24 GB cards, `pipeline-parallel-size=3` plus a small
  `--cpu-offload-gb` budget is the practical way to fit the model.

Use `scripts/nanu/vllm-tp-up.sh` for this shape:

```bash
ssh nanu 'VLLM_TP=1 VLLM_PP=3 VLLM_CPU_OFFLOAD_GB=3 VLLM_MAX_MODEL_LEN=8192 \
  /home/aaron/nanu-scripts/vllm-tp-up.sh openai/gpt-oss-120b gpt-oss:120b'

uv run llm-batch-pipeline run \
    --batch-dir <batch> \
    --backend vllm \
    --base-url http://nanu:18000 \
    --num-shards 1 \
    --num-parallel-jobs 1 \
    --model gpt-oss:120b \
    --plugin spam_detection \
    --request-timeout 1800 \
    --auto-approve

ssh nanu /home/aaron/nanu-scripts/vllm-tp-down.sh
```

This is the topology used by `docs/benchmark-run.md` for the live
`gpt-oss:120b` benchmark on `nanu`.

## Internal data-parallel deployment

vLLM also supports a single-endpoint data-parallel deployment via
`--data-parallel-size N`. The pipeline sees this as a single base URL and lets
vLLM round-robin internally:

```bash
vllm serve <model> --data-parallel-size 4 --host 0.0.0.0 --port 8000
```

This trades fine-grained per-shard observability for simpler client config.

## Telemetry and Grafana

The vLLM backend emits **identical** structured log fields and Prometheus
metric labels as the Ollama backend, with `backend="vllm"` in place of
`backend="ollama"`. All existing Grafana dashboards work; add a `backend`
variable filter (`{backend=~"ollama|vllm|llamacpp"}`) to compare side by side.

## Reproducibility

- **Same engine, same seed, same temperature=0** → bit-identical outputs.
- **Different engines (Ollama vs. vLLM)** are **not** bit-identical even at
  `temperature=0` because they use different inference kernels, tokenizer
  paths, and quantization. Expect high label agreement on classification
  tasks (≥ 70% on the 500-sample spam benchmark, often much higher) but not
  byte-perfect matches.

## See also

- `docs/benchmark-run.md` — the canonical Ollama-vs-vLLM benchmarking runbook.
- `scripts/nanu/README.md` — the helper scripts for sequential bring-up.
