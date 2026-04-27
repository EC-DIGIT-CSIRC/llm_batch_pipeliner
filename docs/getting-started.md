# Getting Started

This is the shortest end-to-end walkthrough for a new user who wants to see what `llm-batch-pipeline` does in practice.

It covers three real workflows:

- OpenAI Batch API with `gpt-4o-mini`
- A 3-way sharded Ollama setup at `http://nanu:11435`, `http://nanu:11436`, and `http://nanu:11437` (Note: "nanu" is my server name for the ollama server. Replace by your ollama server).
- A native llama.cpp server on `nanu` at `http://nanu:18100`

These instructions were tested against the live services on 2026-04-27, including llama.cpp on `nanu`.

## What You Will Do

You will:

- create a batch job with the built-in `spam_detection` plugin
- add two sample `.eml` files
- render a batch JSONL file
- submit it to a backend
- validate the model output against a Pydantic schema
- evaluate the predictions against ground truth

## Prerequisites

- Python 3.13+
- `uv`
- dependencies installed:

```bash
uv sync
```

- for OpenAI: a `.env` file in the repo root with `OPENAI_API_KEY=...`

The CLI will auto-load `.env` from the repository root.

## Offline Sanity Check

Before using any backend, please verify the install:

```bash
uv run llm-batch-pipeline list      # list the plugins
uv sync --group dev
uv run pytest -q                    # quick self-test. if this fails, please submit a bug report
```

## OpenAI Batch Walkthrough

### 1. Create a batch directory

```bash
uv run llm-batch-pipeline init getting_started_openai --plugin spam_detection --model gpt-4o-mini
```

This creates a directory like `batches/batch_001_getting_started_openai`.
Use that path in the commands below as `<openai-batch-dir>`.

### 2. Copy the built-in prompt and schema into the batch

```bash
cp src/llm_batch_pipeline/examples/spam_detection/prompt.txt <openai-batch-dir>/prompt.txt
cp src/llm_batch_pipeline/examples/spam_detection/schema.py <openai-batch-dir>/schema.py
```

If you feel adventurous, you can modify the prompt.txt. Note that it needs to fit to schema.py of course. `schema.py` is a pydantic class which helps in validating the answers that the LLM sends back. Of course, every field which is mentioned in schema.py must be present in the `prompt.txt` and vice versa. It really helps to test out the prompt and the schema on single files.

### 3. Add two sample emails

```bash
cat > <openai-batch-dir>/input/ham__team_sync.eml <<'EOF'
From: alice@example.com
To: bob@example.com
Subject: Team sync tomorrow
Date: Mon, 1 Jan 2024 10:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Hi Bob,

Can we meet tomorrow at 3pm to review the release checklist and assign the last two action items?

Thanks,
Alice
EOF

cat > <openai-batch-dir>/input/spam__million_prize.eml <<'EOF'
From: prizes@claim-now.biz
To: victim@example.com
Subject: URGENT!! Claim your 1000000 dollar prize now
Date: Mon, 1 Jan 2024 11:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Congratulations!

You have been selected to receive a 1000000 dollar cash prize. Click http://claim-prize-now.example.com immediately and send your bank details today to avoid losing your winnings.
EOF
```

### 4. Add a category map for evaluation

```bash
cat > <openai-batch-dir>/evaluation/category-map.json <<'EOF'
{
  "ham": "ham",
  "spam": "spam"
}
EOF
```

The `ham__...` and `spam__...` filename prefixes are how the evaluator infers ground truth from this file.

### 5. Render the batch JSONL

```bash
uv run llm-batch-pipeline render --batch-dir <openai-batch-dir> --plugin spam_detection
```

This writes the request payload to `<openai-batch-dir>/job/batch-00001.jsonl`.


This should give something like this:
```
- discover ok: Discovered 2 files
  discover: completed — 2 files
  filter_1: completed — kept 2/2
- filter_1 ok: filter_1: kept 2/2
  transform: completed — 2 files
⠋ Filtering (post) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/2   0% 0:00:00 < -:--:-- ?it/s- transform ok: transform: transformed 2 files
  filter_2: completed — kept 2/2
  render: completed — 1 shard(s), 2 requests

Rendered 1 shard(s) to (...)/llm_batch_pipeliner/<openai-batch-dir>/job
- render ok: Rendered 2 requests into 1 shard(s)
```

You can now look at the file. It is essentially a JSONL file suitable for submission to openai's Batch API:
```bash
jq -C .  <openai-batch-dir>/job/batch-00001.jsonl
```
It's interesting to see how the prompt as well as the rendered schema.py end up as a list of JSON structures for openai.


Next it's time to ...

### 6. Submit to OpenAI Batch API

```bash
uv run llm-batch-pipeline submit --batch-dir <openai-batch-dir> --backend openai
```

Notes:

- this command waits for the batch to complete by default
- in the live test for this guide, a 2-request batch took about 45 minutes to finish
- batch metadata is saved to `<openai-batch-dir>/output/submission.json`

If you do not want to keep the terminal open:

```bash
uv run llm-batch-pipeline submit --batch-dir <openai-batch-dir> --backend openai --no-wait
uv run llm-batch-pipeline submit --batch-dir <openai-batch-dir> --backend openai --resume-batch-id <batch-id>
```

You can also go to [platform.openai.com](https://platform.openai.com), log in and go to the batches tab:

<img width="1483" height="616" alt="image" src="https://github.com/user-attachments/assets/56d4c9cb-028d-473f-857d-aa43975711c6" />

It can take 24h to process the batch file. You can't specify shorter time to completion windows than 24h. 
After the batch completed, the system can download the resulting output and ...

### 7. Validate the output

```bash
uv run llm-batch-pipeline validate --batch-dir <openai-batch-dir>
```

This reads `<openai-batch-dir>/output/output.jsonl` and writes validated rows to `<openai-batch-dir>/results/validated.json`.

### 8. Evaluate the predictions

```bash
uv run llm-batch-pipeline evaluate \
  --batch-dir <openai-batch-dir> \
  --label-field classification \
  --confidence-field confidence \
  --positive-class spam
```

This prints accuracy, macro F1, per-class metrics, and the confusion matrix to the terminal.

In the tested run, the OpenAI batch classified both sample emails correctly.

## Ollama Walkthrough

### 1. Create a batch directory

```bash
uv run llm-batch-pipeline init getting_started_ollama --plugin spam_detection --model gemma4:latest
```

This creates a directory like `batches/batch_002_getting_started_ollama`.
Use that path in the commands below as `<ollama-batch-dir>`.

### 2. Copy the built-in prompt and schema into the batch

```bash
cp src/llm_batch_pipeline/examples/spam_detection/prompt.txt <ollama-batch-dir>/prompt.txt
cp src/llm_batch_pipeline/examples/spam_detection/schema.py <ollama-batch-dir>/schema.py
```

### 3. Add the same sample inputs and evaluation map

```bash
cat > <ollama-batch-dir>/input/ham__team_sync.eml <<'EOF'
From: alice@example.com
To: bob@example.com
Subject: Team sync tomorrow
Date: Mon, 1 Jan 2024 10:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Hi Bob,

Can we meet tomorrow at 3pm to review the release checklist and assign the last two action items?

Thanks,
Alice
EOF

cat > <ollama-batch-dir>/input/spam__million_prize.eml <<'EOF'
From: prizes@claim-now.biz
To: victim@example.com
Subject: URGENT!! Claim your 1000000 dollar prize now
Date: Mon, 1 Jan 2024 11:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Congratulations!

You have been selected to receive a 1000000 dollar cash prize. Click http://claim-prize-now.example.com immediately and send your bank details today to avoid losing your winnings.
EOF

cat > <ollama-batch-dir>/evaluation/category-map.json <<'EOF'
{
  "ham": "ham",
  "spam": "spam"
}
EOF
```

### 4. Render the batch JSONL

```bash
uv run llm-batch-pipeline render --batch-dir <ollama-batch-dir> --plugin spam_detection
```

### 5. Submit to the 3-way sharded Ollama cluster

```bash
uv run llm-batch-pipeline submit \
  --batch-dir <ollama-batch-dir> \
  --backend ollama \
  --model gemma4:latest \
  --base-url http://nanu:11435 \
  --base-url http://nanu:11436 \
  --base-url http://nanu:11437 \
  --num-shards 3 \
  --num-parallel-jobs 1
```

Notes:

- these exact three URLs were verified for this guide
- `http://11436` is not a valid endpoint; use `http://nanu:11436`
- in the live test for this guide, the full 2-request Ollama submission finished in about 6 seconds

### 6. Validate the output

```bash
uv run llm-batch-pipeline validate --batch-dir <ollama-batch-dir>
```

### 7. Evaluate the predictions

```bash
uv run llm-batch-pipeline evaluate \
  --batch-dir <ollama-batch-dir> \
  --label-field classification \
  --confidence-field confidence \
  --positive-class spam
```

In the tested run, the Ollama batch also classified both sample emails correctly.

## llama.cpp Walkthrough

### 1. Create a batch directory

```bash
uv run llm-batch-pipeline init getting_started_llamacpp --plugin spam_detection --model qwen3.6:latest
```

This creates a directory like `batches/batch_003_getting_started_llamacpp`.
Use that path in the commands below as `<llamacpp-batch-dir>`.

### 2. Copy the built-in prompt and schema into the batch

```bash
cp src/llm_batch_pipeline/examples/spam_detection/prompt.txt <llamacpp-batch-dir>/prompt.txt
cp src/llm_batch_pipeline/examples/spam_detection/schema.py <llamacpp-batch-dir>/schema.py
```

### 3. Add the same sample inputs and evaluation map

Use the same `ham__team_sync.eml`, `spam__million_prize.eml`, and
`evaluation/category-map.json` blocks from the Ollama walkthrough above.

### 4. Render the batch JSONL

```bash
uv run llm-batch-pipeline render --batch-dir <llamacpp-batch-dir> --plugin spam_detection
```

### 5. Start native llama.cpp

```bash
ssh nanu 'sudo systemctl stop ollama'
ssh nanu 'LLAMACPP_VISIBLE_DEVICES=0,1,2 LLAMACPP_DEVICE_LIST=CUDA0,CUDA1,CUDA2 LLAMACPP_PORT=18100 LLAMACPP_SPLIT_MODE=layer LLAMACPP_TENSOR_SPLIT=1,1,1 LLAMACPP_N_GPU_LAYERS=all LLAMACPP_CTX_SIZE=32768 LLAMACPP_ENDPOINT=chat /home/aaron/nanu-scripts/llamacpp-up.sh /data/models/llama.cpp/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf qwen3.6:latest'
```

This guide uses the fresh Hugging Face GGUF under `/data/models/llama.cpp/`.
The server should answer `http://nanu:18100/v1/models`.

### 6. Submit to llama.cpp

```bash
uv run llm-batch-pipeline submit --batch-dir <llamacpp-batch-dir> --backend llamacpp --llamacpp-endpoint chat --base-url http://nanu:18100 --num-shards 1 --num-parallel-jobs 1 --model qwen3.6:latest --request-timeout 1200
```

### 7. Validate the output

```bash
uv run llm-batch-pipeline validate --batch-dir <llamacpp-batch-dir>
```

### 8. Evaluate the predictions

```bash
uv run llm-batch-pipeline evaluate \
  --batch-dir <llamacpp-batch-dir> \
  --label-field classification \
  --confidence-field confidence \
  --positive-class spam
```

### 9. Stop llama.cpp and restore Ollama

```bash
ssh nanu '/home/aaron/nanu-scripts/llamacpp-down.sh'
ssh nanu 'sudo systemctl start ollama'
```

In the tested run, the llama.cpp batch also classified both sample emails correctly.

## Output You Should Expect

After `render`:

- `<batch-dir>/job/batch-00001.jsonl`

After `submit`:

- `<batch-dir>/output/output.jsonl`
- `<batch-dir>/output/summary.json`

After `validate`:

- `<batch-dir>/results/validated.json`

After `evaluate`:

- metrics printed to stdout

## Other backends

The pipeline supports three submission backends, all sharing the same CLI flags:

- `--backend openai` (default) — OpenAI Batch API
- `--backend ollama` — local Ollama servers; `--base-url` repeatable for sharding
- `--backend llamacpp` — local native `llama-server`; use `--llamacpp-endpoint chat` and `--base-url http://HOST:18100`
- `--backend vllm` — local `vllm serve` instances; `--base-url` repeatable for
  sharding; optional `--api-key` / `VLLM_API_KEY`; optional
  `--vllm-endpoint chat` to use `/v1/chat/completions` instead of the default
  `/v1/responses`

For a vLLM-specific walkthrough including 3-way sharding, see
[`docs/running-vllm.md`](running-vllm.md). For sequential
Ollama-vs-vLLM benchmarking on the same hardware, see
[`docs/benchmark-run.md`](benchmark-run.md).
For a native llama.cpp walkthrough, see [`docs/running-llamacpp.md`](running-llamacpp.md).

## When To Use `run` Instead

If you already trust your prompt, schema, and backend settings, you can collapse the whole pipeline into one command:

```bash
uv run llm-batch-pipeline run --batch-dir <batch-dir> --plugin spam_detection --auto-approve ...
```

For a first pass, the staged workflow above is easier to debug because you can inspect the rendered JSONL, raw model output, validated JSON, and evaluation step separately.
