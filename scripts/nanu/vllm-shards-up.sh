#!/usr/bin/env bash
# Spin up 3 vLLM shard servers on ports 18000/18001/18002, pinned to GPU 0/1/2.
# Each shard is a separate `vllm serve` process (tensor-parallel-size 1).
#
# Usage:
#   vllm-shards-up.sh <model_weights> [served_model_name]
#
# Examples:
#   # Ollama-style name maps to vLLM-side HF weights:
#   vllm-shards-up.sh google/gemma-2-9b-it gemma4:latest
#
#   # Use the same name on both sides:
#   vllm-shards-up.sh google/gemma-2-9b-it
#
# Preconditions:
#   - Ollama shards on 11435/11436/11437 must be DOWN (run ollama-shards-down.sh first).
#   - `vllm` binary must be on PATH for the user invoking the script.
#
# Environment:
#   VLLM_PYTHON                   default: $HOME/git/vllm/.venv/bin/python
#                                 must be a Python interpreter that has vLLM importable
#   VLLM_API_KEY                  optional bearer token to require on the server side
#   VLLM_GPU_MEMORY_UTILIZATION   default: 0.9
#   VLLM_MAX_MODEL_LEN            optional integer
#   VLLM_DTYPE                    default: auto
#   VLLM_EXTRA_ARGS               extra flags appended verbatim to `vllm serve`
#   WAIT_TIMEOUT_S                default: 600 (first start can take a while)

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <model_weights> [served_model_name]" >&2
  exit 2
fi

MODEL_WEIGHTS="$1"
SERVED_NAME="${2:-$MODEL_WEIGHTS}"

VLLM_PYTHON="${VLLM_PYTHON:-$HOME/git/vllm/.venv/bin/python}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-600}"
LOG_DIR="/tmp/vllm-shards"

SHARDS=(
  "18000:0"
  "18001:1"
  "18002:2"
)

OLLAMA_PORTS=(11435 11436 11437)

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Preconditions ---
for port in "${OLLAMA_PORTS[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"; then
    die "Ollama shard on port ${port} is still up. Run ollama-shards-down.sh first."
  fi
done

for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"; then
    die "Port ${port} is already in use. Run vllm-shards-down.sh first."
  fi
done

[ -x "$VLLM_PYTHON" ] || die "VLLM_PYTHON=$VLLM_PYTHON does not exist or is not executable"
"$VLLM_PYTHON" -c "import vllm" 2>/dev/null || die "vllm not importable from $VLLM_PYTHON"

mkdir -p "$LOG_DIR"

# Assemble optional flags
extra_flags=()
if [ -n "$VLLM_API_KEY" ]; then
  extra_flags+=(--api-key "$VLLM_API_KEY")
fi
if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
  extra_flags+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi
# shellcheck disable=SC2206
extra_verbatim=($VLLM_EXTRA_ARGS)

# --- Start shards ---
for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  gpu="${spec##*:}"
  log_file="$LOG_DIR/vllm-${port}.log"
  log "Starting vLLM shard port=${port} gpu=${gpu} model=${MODEL_WEIGHTS} served_as=${SERVED_NAME} log=${log_file}"

  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$VLLM_PYTHON" -c 'from vllm.entrypoints.cli.main import main; main()' serve "$MODEL_WEIGHTS" \
      --host 0.0.0.0 \
      --port "$port" \
      --tensor-parallel-size 1 \
      --served-model-name "$SERVED_NAME" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --dtype "$VLLM_DTYPE" \
      "${extra_flags[@]}" \
      "${extra_verbatim[@]}" \
      >"$log_file" 2>&1 </dev/null &
done

# --- Wait for /v1/models on each shard ---
auth_header=()
if [ -n "$VLLM_API_KEY" ]; then
  auth_header=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  waited=0
  while true; do
    if curl -fsS -o /dev/null "${auth_header[@]}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null; then
      log "Shard ${port} is answering /v1/models"
      break
    fi
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
      die "Shard ${port} failed to answer /v1/models within ${WAIT_TIMEOUT_S}s. See $LOG_DIR/vllm-${port}.log"
    fi
  done
done

log "All 3 vLLM shards are up: 18000 18001 18002 (served as ${SERVED_NAME})"
