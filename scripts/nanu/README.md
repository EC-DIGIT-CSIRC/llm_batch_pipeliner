# nanu benchmark scripts

Sequential 3-way sharded benchmarking helpers for `nanu`. Each script is
self-contained and exits non-zero on failure.

## Layout

| Script                  | Purpose                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| `ollama-shards-up.sh`   | Start 3 Ollama shards on `:11435/:11436/:11437`, pinned to GPU 0/1/2, `OLLAMA_NOPRUNE=true` enforced.          |
| `ollama-shards-down.sh` | Stop those 3 shards. Does **not** touch the system `ollama.service` on `:11434`.                               |
| `ollama-tp-up.sh`       | Start one temporary Ollama server on `:11440`, restricted to a chosen GPU set (default GPUs 0,1).              |
| `ollama-tp-down.sh`     | Stop that one temporary Ollama TP-style server.                                                                  |
| `vllm-shards-up.sh`     | Start 3 `vllm serve` shards on `:18000/:18001/:18002`, pinned to GPU 0/1/2.                                     |
| `vllm-shards-down.sh`   | Stop those 3 vLLM shards.                                                                                       |
| `vllm-tp-up.sh`         | Start one vLLM server on `:18000` spanning GPUs 0/1/2 with tensor/pipeline parallelism.                        |
| `vllm-tp-down.sh`       | Stop that one TP/PP vLLM server.                                                                                |
| `llamacpp-up.sh`        | Start one native `llama-server` on `:18100` across a chosen GPU set using a GGUF in `/data/models`.            |
| `llamacpp-down.sh`      | Stop that native `llama-server`.                                                                                |
| `gpu-idle-check.sh`     | Assert GPUs 0/1/2 have `< MAX_USED_MIB` (default 500) before launching the next engine.                         |

## Why sequential?

Large models such as `gpt-oss:120b` take effectively all the available GPU
memory on `nanu`. Running both engines simultaneously on the same GPUs causes
OOMs and skews speed numbers.
Tear one down, wait until VRAM clears, then bring the other up.

## Install on nanu

These scripts live in the repo at `scripts/nanu/` and are scp-copied to nanu:

```bash
ssh nanu 'mkdir -p /home/aaron/nanu-scripts'
scp scripts/nanu/*.sh scripts/nanu/README.md nanu:/home/aaron/nanu-scripts/
ssh nanu 'chmod +x /home/aaron/nanu-scripts/*.sh'
```

## Standard benchmark sequence

See `docs/benchmark-run.md` for the full runbook.

```bash
# Ollama side (gpt-oss uses the system ollama.service on :11434)
uv run llm-batch-pipeline run --batch-dir <ollama-dir> --backend ollama \
    --base-url http://nanu:11434 \
    --model gpt-oss:120b --num-shards 1 --num-parallel-jobs 4 --auto-approve
ssh nanu 'sudo systemctl stop ollama'
ssh nanu /home/aaron/nanu-scripts/gpu-idle-check.sh

# vLLM side (same public model, different topology)
ssh nanu 'VLLM_TP=1 VLLM_PP=3 VLLM_CPU_OFFLOAD_GB=3 VLLM_MAX_MODEL_LEN=8192 \
  /home/aaron/nanu-scripts/vllm-tp-up.sh openai/gpt-oss-120b gpt-oss:120b'
uv run llm-batch-pipeline run --batch-dir <vllm-dir> --backend vllm \
    --base-url http://nanu:18000 \
    --model gpt-oss:120b --num-shards 1 --num-parallel-jobs 1 --auto-approve
ssh nanu /home/aaron/nanu-scripts/vllm-tp-down.sh
ssh nanu 'sudo systemctl start ollama'
```

## Alternative: qwen3.6 tensor-parallel / no-sharding benchmark

This mode uses **one endpoint** per backend and a **shared two-GPU** allocation
(`CUDA_VISIBLE_DEVICES=0,1`), which is the closest no-sharding analogue for
`qwen3.6:latest` on `nanu`.

```bash
# Prepare a deterministic 300-mail balanced subset (180 ham / 120 spam)
scripts/make_spam_benchmark_subset.py \
  batches/batch_009_spam_benchmark_3shard_dashboard_repeat \
  batches/batch_030_qwen3_6_tp_300 \
  --ham 180 --spam 120 --seed 42

# Ollama side
ssh nanu 'sudo systemctl stop ollama'
ssh nanu /home/aaron/nanu-scripts/gpu-idle-check.sh
ssh nanu 'OLLAMA_VISIBLE_DEVICES=0,1 OLLAMA_PORT=11440 /home/aaron/nanu-scripts/ollama-tp-up.sh qwen3.6:latest'
uv run llm-batch-pipeline run --batch-dir batches/batch_030_qwen3_6_tp_300 --backend ollama \
  --base-url http://nanu:11440 --num-shards 1 --num-parallel-jobs 1 --model qwen3.6:latest --plugin spam_detection --auto-approve
ssh nanu 'OLLAMA_PORT=11440 /home/aaron/nanu-scripts/ollama-tp-down.sh'
ssh nanu 'sudo systemctl start ollama'
ssh nanu /home/aaron/nanu-scripts/gpu-idle-check.sh GPUS=0,1

# vLLM side
ssh nanu 'VLLM_PYTHON=$HOME/git/vllm/.venv/bin/python VLLM_VISIBLE_DEVICES=0,1 VLLM_PORT=18040 VLLM_TP=2 VLLM_PP=1 VLLM_MAX_MODEL_LEN=32768 VLLM_EXTRA_ARGS="--enable-expert-parallel --reasoning-parser qwen3 --language-model-only" /home/aaron/nanu-scripts/vllm-tp-up.sh QuantTrio/Qwen3.6-35B-A3B-AWQ qwen3.6:latest'
uv run llm-batch-pipeline run --batch-dir batches/batch_031_qwen3_6_tp_300 --backend vllm \
  --base-url http://nanu:18040 --num-shards 1 --num-parallel-jobs 1 --model qwen3.6:latest --plugin spam_detection --auto-approve
ssh nanu 'VLLM_PORT=18040 /home/aaron/nanu-scripts/vllm-tp-down.sh'
```

## llama.cpp native benchmark

This uses the fresh Hugging Face `qwen3.6:latest` GGUF stored under
`/data/models/llama.cpp/` and runs `llama-server` natively on `nanu`.

```bash
# Build once on nanu (from a local clone of https://github.com/ggml-org/llama.cpp)
cmake -B /home/aaron/git/llama.cpp/build -DGGML_CUDA=ON
cmake --build /home/aaron/git/llama.cpp/build --config Release -t llama-server

# Run one native llama.cpp server
LLAMACPP_ENDPOINT=chat /home/aaron/nanu-scripts/llamacpp-up.sh \
  /data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  qwen3.6:latest

# Stop it
/home/aaron/nanu-scripts/llamacpp-down.sh
```

## Safety notes

- `ollama-shards-up.sh` always sets `OLLAMA_NOPRUNE=true`. **Never bypass this.**
  Ollama will otherwise prune unreferenced blobs from the shared
  `/data/models/ollama` store on startup. See repo `AGENTS.md`.
- `vllm-shards-up.sh` aborts if the Ollama shard ports are still bound.
- `gpu-idle-check.sh` is the cheap insurance against starting vLLM before
  Ollama has released VRAM.
- `llamacpp-up.sh` uses the fresh `/data/models/llama.cpp` GGUF directly and
  does not touch the Ollama store, so it is safe as long as the file remains in
  place.
