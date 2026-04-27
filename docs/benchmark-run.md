# Benchmark runbook: Ollama vs vLLM, sequential

The goal of this runbook is to compare **inference speed** of two local
backends (Ollama and vLLM) on identical hardware (`nanu`), identical inputs,
and the **same stable public model family**:

- Ollama: `qwen3:8b`
- vLLM: `Qwen/Qwen3-8B`, served as `qwen3:8b`

## Why sequential, not concurrent

Running both engines simultaneously on the same GPUs causes VRAM contention
and skews speed measurements. The protocol is therefore:

1. Bring **Ollama** up
2. Run the batch
3. Bring Ollama down
4. **Wait until GPUs are idle** (`gpu-idle-check.sh`)
5. Bring **vLLM** up
6. Run the same batch (fresh batch directory)
7. Bring vLLM down
8. Compare

## Scripts on `nanu`

All helpers live in this repo at `scripts/nanu/` and are scp-copied to
`/home/aaron/nanu-scripts/` on `nanu`:

| Script                  | Purpose                                                            |
|-------------------------|--------------------------------------------------------------------|
| `ollama-shards-up.sh`   | Start 3 Ollama shards on 11435/11436/11437 (GPU 0/1/2). Forces `OLLAMA_NOPRUNE=true`. |
| `ollama-shards-down.sh` | Stop those 3 shards. Leaves system `ollama.service` on 11434 alone. |
| `vllm-shards-up.sh`     | Start 3 independent vLLM shards on 18000/18001/18002 (GPU 0/1/2).   |
| `vllm-shards-down.sh`   | Stop those 3 vLLM shards.                                            |
| `vllm-tp-up.sh`         | Start one vLLM server on 18000 using all 3 GPUs via TP/PP.           |
| `vllm-tp-down.sh`       | Stop that one TP/PP vLLM server.                                     |
| `gpu-idle-check.sh`     | Assert GPUs 0/1/2 use `<MAX_USED_MIB>` (default 500 MiB) of VRAM.   |

Install or refresh:

```bash
ssh nanu 'mkdir -p /home/aaron/nanu-scripts'
scp scripts/nanu/*.sh scripts/nanu/README.md nanu:/home/aaron/nanu-scripts/
ssh nanu 'chmod +x /home/aaron/nanu-scripts/*.sh'
```

## Prerequisites

- `qwen3:8b` already pulled into Ollama: `ssh nanu 'ollama list | grep qwen3:8b'`
- `Qwen/Qwen3-8B` resolvable from HF (first vLLM shard startup downloads it).
- vLLM installed in `~/git/vllm/.venv` on `nanu`.
- Both backends run as **true 3-way sharded one-model-per-GPU** setups on
  the same 3 RTX 4090 GPUs.

## Step 1 — Ollama benchmark

```bash
# Bring up Ollama shards
ssh nanu /home/aaron/nanu-scripts/ollama-shards-up.sh

# Run the benchmark batch
uv run llm-batch-pipeline run \
    --batch-dir batches/batch_NNN_spam_benchmark_ollama \
    --backend ollama \
    --base-url http://nanu:11435 \
    --base-url http://nanu:11436 \
    --base-url http://nanu:11437 \
    --num-shards 3 \
    --num-parallel-jobs 1 \
    --model qwen3:8b \
    --plugin spam_detection \
    --request-timeout 600 \
    --auto-approve

# Tear down + idle check
ssh nanu /home/aaron/nanu-scripts/ollama-shards-down.sh
ssh nanu /home/aaron/nanu-scripts/gpu-idle-check.sh
```

Record the printed `service_run_key` — you will need it for Grafana filtering.

## Step 2 — vLLM benchmark

```bash
# Bring up vLLM shards (three independent one-GPU servers)
ssh nanu 'VLLM_PYTHON=$HOME/git/vllm/.venv/bin/python VLLM_MAX_MODEL_LEN=32768 /home/aaron/nanu-scripts/vllm-shards-up.sh Qwen/Qwen3-8B qwen3:8b'

# Run the benchmark on a *fresh* batch directory (same inputs)
uv run llm-batch-pipeline run \
    --batch-dir batches/batch_NNN_spam_benchmark_vllm \
    --backend vllm \
    --base-url http://nanu:18000 \
    --base-url http://nanu:18001 \
    --base-url http://nanu:18002 \
    --num-shards 3 \
    --num-parallel-jobs 1 \
    --model qwen3:8b \
    --plugin spam_detection \
    --request-timeout 600 \
    --auto-approve

# Tear down
ssh nanu /home/aaron/nanu-scripts/vllm-shards-down.sh
```

## Step 3 — Verify telemetry in Bee

```bash
# Loki: the row_summary log for both runs (different service_run_key)
curl -fsS --get "https://bee.lo-res.org/loki/api/v1/query" \
    --data-urlencode 'query=last_over_time(
        {service_name="llm-batch-pipeline"}
        | json
        | status="row_summary"
        | logger=~"llm_batch_pipeline.backends.(ollama|vllm|llamacpp)"
        | unwrap row_duration_p50_ms [24h]
    )'

# Prometheus: per-backend request counters with the backend label
curl -fsS --get "https://bee.lo-res.org/prometheus/api/v1/query" \
    --data-urlencode 'query=sum by (backend, status) (
        last_over_time(llm_batch_pipeline_requests_total[24h])
    )'
```

You should see both `backend="ollama"` and `backend="vllm"` series.

## Step 4 — Read the comparison

Two equivalent paths:

### A. From local artifacts

Each batch directory contains:

- `output/summary.json` with `"requests_per_second"` and friends
- `logs/metrics.json` with stage durations
- `logs/pipeline.jsonl` with the `row_summary` event (avg/p50/min/max ms)

A small helper (TBD; tracked in follow-up) writes
`comparison.json` from a matched run pair:

```json
{
  "model": "qwen3:8b",
  "sample_size": 100,
  "ollama": {"requests_per_second": ..., "row_duration_p50_ms": ..., "accuracy": ...},
  "vllm":   {"requests_per_second": ..., "row_duration_p50_ms": ..., "accuracy": ...},
  "label_agreement_rate": ...
}
```

### B. From Grafana

The existing dashboards have a `Run Key` selector. Open two browser tabs (one
per run key) for side-by-side comparison, or use the dedicated 3-way backend
comparison dashboard to compare `ollama`, `vllm`, and `llamacpp` together.

## Reproducibility caveats

- **Same engine + same seed + temperature=0** → bit-identical outputs.
- **Different engines** are NOT bit-identical at `temperature=0`. Different
  kernels, tokenizer trip points, and quantization codepaths cause divergence.
- Expected label-level agreement on the spam benchmark: **≥ 70%**, usually
  significantly higher.

- Note on *speed* comparisons: the topology differs slightly because of VRAM
  constraints.
  - Ollama: one service on `:11434`, internally spanning all 3 GPUs.
  - vLLM: one server on `:18000`, `pipeline-parallel-size=3`, `cpu-offload-gb=3`.
  This is still a fair same-host comparison, but not a same-client-sharding
  comparison. The benchmark runbook deliberately captures the actual deployment
  shapes for each engine.

## Live result on `nanu` (recommended stable setup)

The current recommended stable benchmark pair on `nanu` uses `qwen3:8b` and a
matched **100-sample** subset of the SpamAssassin corpus.

Matched 100-sample comparison (`batches/qwen3_8b_100_comparison_backend.json`):

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

- vLLM is **faster** on this host for this model and batch:
  - throughput: `0.851 req/s` vs `0.504 req/s`
  - p50 row wall-clock: `3369 ms` vs `5452 ms`
- Label agreement across the two engines on this prompt/model pair is `0.73`
  over the same 100 inputs.

## Advanced / experimental note: `gpt-oss:120b`

`gpt-oss:120b` can be benchmarked on `nanu`, but it is **not** the
recommended default comparison target on this hardware.

- Ollama can serve it via the system `ollama.service` on `:11434`.
- vLLM requires a patched startup path (`pipeline-parallel-size=3`, CPU
  offload, and a workaround for the CPU-offload reinitialization assertion).
- Even after those workarounds, the full 100-sample vLLM run was unstable; the
  stable matched comparison we obtained was a **28-sample** subset.

Use `qwen3:8b` for day-to-day backend speed comparisons. Reserve
`gpt-oss:120b` for targeted large-model experiments.

## Alternative plan: 300-mail `qwen3.6` benchmark, tensor-parallel mode, no sharding

If you specifically want a **no-sharding** comparison for a larger model, the
recommended setup on `nanu` is:

- Ollama: `qwen3.6:latest`
- vLLM: `QuantTrio/Qwen3.6-35B-A3B-AWQ`, served as `qwen3.6:latest`
- GPUs: `0,1` only
- One endpoint per backend

### Why 2 GPUs and not 3?

For this benchmark we want **tensor-parallel mode, no sharding**.
The cleanest stable shape is therefore one server per backend using a chosen
GPU set, not multiple shard endpoints.

- Ollama side: one temporary server on `:11440`, restricted to GPUs 0 and 1
- vLLM side: one TP=2 server on `:18040`, restricted to GPUs 0 and 1

GPU 2 remains idle.

### Prepare the deterministic 300-mail subset

Use the helper script to create a balanced benchmark subset:

```bash
scripts/make_spam_benchmark_subset.py \
  batches/batch_009_spam_benchmark_3shard_dashboard_repeat \
  batches/batch_030_qwen3_6_tp_300 \
  --ham 180 --spam 120 --seed 42
```

Then copy it for the vLLM side:

```bash
cp -R batches/batch_030_qwen3_6_tp_300 batches/batch_031_qwen3_6_tp_300
```

### Ollama side

```bash
ssh nanu 'sudo systemctl stop ollama'
ssh nanu 'GPUS=0,1 /home/aaron/nanu-scripts/gpu-idle-check.sh'

ssh nanu 'OLLAMA_VISIBLE_DEVICES=0,1 OLLAMA_PORT=11440 /home/aaron/nanu-scripts/ollama-tp-up.sh qwen3.6:latest'

uv run llm-batch-pipeline run \
  --batch-dir batches/batch_030_qwen3_6_tp_300 \
  --backend ollama \
  --base-url http://nanu:11440 \
  --num-shards 1 \
  --num-parallel-jobs 1 \
  --model qwen3.6:latest \
  --plugin spam_detection \
  --request-timeout 1200 \
  --auto-approve

ssh nanu 'OLLAMA_PORT=11440 /home/aaron/nanu-scripts/ollama-tp-down.sh'
ssh nanu 'sudo systemctl start ollama'
ssh nanu 'GPUS=0,1 /home/aaron/nanu-scripts/gpu-idle-check.sh'
```

### vLLM side

The currently recommended vLLM weights are the AWQ variant because the base
BF16 weights are too large for 2×24 GB.

```bash
ssh nanu 'VLLM_PYTHON=$HOME/git/vllm/.venv/bin/python VLLM_VISIBLE_DEVICES=0,1 VLLM_PORT=18040 VLLM_TP=2 VLLM_PP=1 VLLM_MAX_MODEL_LEN=32768 VLLM_GPU_MEMORY_UTILIZATION=0.9 VLLM_EXTRA_ARGS="--enable-expert-parallel --reasoning-parser qwen3 --language-model-only" /home/aaron/nanu-scripts/vllm-tp-up.sh QuantTrio/Qwen3.6-35B-A3B-AWQ qwen3.6:latest'

uv run llm-batch-pipeline run \
  --batch-dir batches/batch_031_qwen3_6_tp_300 \
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

### llama.cpp side

Native llama.cpp uses the fresh Hugging Face GGUF stored under
`/data/models/llama.cpp/`. The recommended benchmark profile is a 3-GPU split
on GPUs 0/1/2 with chat completions.

```bash
ssh nanu 'LLAMACPP_ENDPOINT=chat /home/aaron/nanu-scripts/llamacpp-up.sh \
  /data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  qwen3.6:latest'

uv run llm-batch-pipeline run \
  --batch-dir batches/batch_035_qwen3_6_llamacpp_300 \
  --backend llamacpp \
  --llamacpp-endpoint chat \
  --base-url http://nanu:18100 \
  --num-shards 1 \
  --num-parallel-jobs 1 \
  --model qwen3.6:latest \
  --plugin spam_detection \
  --request-timeout 1200 \
  --auto-approve

ssh nanu '/home/aaron/nanu-scripts/llamacpp-down.sh'
```

### Notes

- This mode is deliberately **not** the default benchmark in this document.
- It is heavier and less battle-tested than the `qwen3:8b` 3-shard comparison.
- If llama.cpp struggles to fit on all three GPUs, fall back to the single-GPU
  smoke profile in `docs/running-llamacpp.md` and keep the comparison note in
  the final artifact.
- If you want strict decoding comparability, a follow-up improvement is to pass
  the same `temperature`, `seed`, and output-token cap explicitly through both
  backends.

## Troubleshooting

| Symptom                          | Fix                                                                     |
|----------------------------------|-------------------------------------------------------------------------|
| `Port XXXX is already in use`    | Run the matching `*-down.sh` first.                                     |
| vLLM shard startup says max seq len too high | Lower `VLLM_MAX_MODEL_LEN` (e.g. `32768`) and restart `vllm-shards-up.sh`. |
| `Ollama shard ... is still up`   | `ollama-shards-down.sh` returned non-zero; check `lsof -i:11435..11437`. |
| Coexistence with `ollama.service` | The system service on `:11434` is independent. Shard ports are 11435/6/7. |
| GPU memory still busy            | `gpu-idle-check.sh` will wait up to `MAX_WAIT_S=60s`; raise if needed.   |

## Safety reminder

The Ollama shard script always sets `OLLAMA_NOPRUNE=true`. **Never bypass it.**
Without it, Ollama will prune unreferenced blobs from the shared
`/data/models/ollama` store on every startup, deleting other users' models.
See `AGENTS.md`.
