# Running Ollama

This document shows how to run `llm-batch-pipeline` against a sharded Ollama setup.

## Critical Safety Rule

If multiple `ollama serve` processes share the same `OLLAMA_MODELS` directory, every shard process must set:

```bash
OLLAMA_NOPRUNE=true
```

Do not start shard servers against a shared model store without it.

## Match the Main Service

Before starting shards, inspect the main Ollama service:

```bash
ssh nanu "systemctl show ollama --property=Environment --no-pager"
```

Carry over the relevant environment variables from the main service, especially:

- `OLLAMA_MODELS`
- `OLLAMA_NUM_PARALLEL`
- `HOME` / runtime user if the main service depends on them

If you add the no-prune setting to a service override, verify the variable name is spelled exactly `OLLAMA_NOPRUNE`.

## Example: 3 Shards on `nanu`

This setup was tested with:

- main service model store: `/data/models/ollama`
- shard ports: `11435`, `11436`, `11437`
- model: `gemma4:latest`

When all shards run on the same multi-GPU host, pin each shard to a dedicated GPU with `CUDA_VISIBLE_DEVICES`.

Start the three shard daemons under the `ollama` user:

```bash
ssh nanu 'sudo mkdir -p /tmp/ollama-shards && sudo chown ollama:ollama /tmp/ollama-shards'

ssh nanu 'sudo -u ollama sh -lc '"'"'HOME=/usr/share/ollama PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11435 OLLAMA_MODELS=/data/models/ollama OLLAMA_NUM_PARALLEL=3 OLLAMA_NOPRUNE=true nohup /usr/local/bin/ollama serve >/tmp/ollama-shards/ollama-11435.log 2>&1 </dev/null &'"'"''

ssh nanu 'sudo -u ollama sh -lc '"'"'HOME=/usr/share/ollama PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11436 OLLAMA_MODELS=/data/models/ollama OLLAMA_NUM_PARALLEL=3 OLLAMA_NOPRUNE=true nohup /usr/local/bin/ollama serve >/tmp/ollama-shards/ollama-11436.log 2>&1 </dev/null &'"'"''

ssh nanu 'sudo -u ollama sh -lc '"'"'HOME=/usr/share/ollama PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11437 OLLAMA_MODELS=/data/models/ollama OLLAMA_NUM_PARALLEL=3 OLLAMA_NOPRUNE=true nohup /usr/local/bin/ollama serve >/tmp/ollama-shards/ollama-11437.log 2>&1 </dev/null &'"'"''
```

Verify that all three ports are listening:

```bash
ssh nanu "ss -ltn | rg ':1143(5|6|7)'"
```

Verify that all three shards can see the shared models:

```bash
curl -fsS http://nanu:11435/api/tags
curl -fsS http://nanu:11436/api/tags
curl -fsS http://nanu:11437/api/tags
```

Inspect shard logs if needed:

```bash
ssh nanu 'tail -n 50 /tmp/ollama-shards/ollama-11435.log'
ssh nanu 'tail -n 50 /tmp/ollama-shards/ollama-11436.log'
ssh nanu 'tail -n 50 /tmp/ollama-shards/ollama-11437.log'
```

## Run the Pipeline Against 3 Shards

Point the pipeline at all three shard URLs. The run below was tested end to end and completed successfully.

```bash
export OTEL_SERVICE_NAME=llm-batch-pipeline
export OTEL_EXPORTER_OTLP_ENDPOINT=http://bee.lo-res.org:4318

uv run llm-batch-pipeline run \
  --batch-dir <batch-dir> \
  --plugin spam_detection \
  --backend ollama \
  --model gemma4:latest \
  --base-url http://nanu:11435 \
  --base-url http://nanu:11436 \
  --base-url http://nanu:11437 \
  --num-shards 3 \
  --num-parallel-jobs 1 \
  --auto-approve
```

Tested result:

- all three shard warmups succeeded
- benchmark run completed `493/493`
- logs and metrics arrived at Bee (`http://bee.lo-res.org:4318`)

## Verify Telemetry

Prometheus on Bee:

```bash
curl -fsS --get https://bee.lo-res.org/prometheus/api/v1/query \
  --data-urlencode 'query=count(llm_batch_pipeline_runs_total{status="completed"})'
```

Loki on Bee:

```bash
curl -fsS --get https://bee.lo-res.org/loki/api/v1/query \
  --data-urlencode 'query=sum(count_over_time({service_name="llm-batch-pipeline"} | json | step="pipeline" | status="completed" [15m]))'
```

Grafana dashboard:

```text
http://grafana-test.intelmq.org:3000/d/llm-batch-pipeline-overview/llm-batch-pipeline-overview
```

## Stop the Shards

```bash
ssh nanu "lsof -t -nP -iTCP:11435 -sTCP:LISTEN 2>/dev/null | xargs -r kill"
ssh nanu "lsof -t -nP -iTCP:11436 -sTCP:LISTEN 2>/dev/null | xargs -r kill"
ssh nanu "lsof -t -nP -iTCP:11437 -sTCP:LISTEN 2>/dev/null | xargs -r kill"
```

## Recommended Practice

If you need shard daemons only temporarily, prefer:

- keeping the main service on `11434`
- starting shard daemons only for the duration of the batch run
- always setting `OLLAMA_NOPRUNE=true` on shard daemons sharing the same model store

If you want full isolation, use a separate `OLLAMA_MODELS` directory per shard instead of sharing one model store.
