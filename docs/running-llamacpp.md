# Running llama.cpp

This document shows how to run `llm-batch-pipeline` against a native
`llama-server` deployment on `nanu`.

## Model storage

Use a fresh Hugging Face GGUF copy under `/data/models/llama.cpp/` instead of
the Ollama blob store:

```text
/data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
```

This copy is pulled from `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` and is the one
that currently loads cleanly with `llama.cpp`.

## Build once on `nanu`

```bash
cd /home/aaron/git/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -t llama-server
```

## 3-GPU benchmark profile

This is the speed-oriented profile for `qwen3.6:latest`.

```bash
LLAMACPP_VISIBLE_DEVICES=0,1,2 \
LLAMACPP_DEVICE_LIST=CUDA0,CUDA1,CUDA2 \
LLAMACPP_PORT=18100 \
LLAMACPP_SPLIT_MODE=layer \
LLAMACPP_TENSOR_SPLIT=1,1,1 \
LLAMACPP_N_GPU_LAYERS=all \
LLAMACPP_CTX_SIZE=32768 \
LLAMACPP_ENDPOINT=chat \
/home/aaron/nanu-scripts/llamacpp-up.sh \
  /data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  qwen3.6:latest
```

Run the pipeline:

```bash
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
```

Stop it:

```bash
/home/aaron/nanu-scripts/llamacpp-down.sh
```

## Single-GPU smoke profile

If you want to sanity-check the server on one GPU first:

```bash
LLAMACPP_VISIBLE_DEVICES=0 \
LLAMACPP_DEVICE_LIST=CUDA0 \
LLAMACPP_SPLIT_MODE=none \
LLAMACPP_TENSOR_SPLIT=1 \
LLAMACPP_N_GPU_LAYERS=all \
LLAMACPP_CTX_SIZE=8192 \
LLAMACPP_ENDPOINT=chat \
/home/aaron/nanu-scripts/llamacpp-up.sh \
  /data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  qwen3.6:latest
```

## Notes

- `llama-server` is OpenAI-compatible; for this Qwen3.6 GGUF, use
  `--llamacpp-endpoint chat` so the schema output validates cleanly.
- Keep `ollama.service` stopped while llama.cpp is using the same GPUs.
- Do not remove or rename the GGUF file under `/data/models/llama.cpp/`.
