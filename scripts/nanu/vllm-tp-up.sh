#!/usr/bin/env bash
# Spin up ONE vLLM server with tensor/pipeline parallelism across a chosen GPU set.
#
# Use this when the model is too big to fit on a single GPU (e.g.
# Qwen/Qwen3.6-35B-A3B AWQ). The pipeline talks to a single endpoint; vLLM
# internally splits the model across GPUs.
#
# Usage:
#   vllm-tp-up.sh <model_weights> [served_model_name]
#
# Examples:
#   vllm-tp-up.sh cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit qwen3.6:latest
#
# Preconditions:
#   - Ollama shards on 11435/11436/11437 must be DOWN (run ollama-shards-down.sh first).
#   - VLLM_PYTHON points to a Python interpreter with vLLM importable.
#
# Environment:
#   VLLM_PYTHON                   default: $HOME/git/vllm/.venv/bin/python
#   VLLM_VISIBLE_DEVICES          default: 0,1,2
#                                 comma-separated CUDA device IDs to expose to the server
#   VLLM_API_KEY                  optional bearer token to require server-side
#   VLLM_PORT                     default: 18000
#   VLLM_TP                       default: 3
#   VLLM_PP                       default: 1 (pipeline-parallel size; raise when TP
#                                 is constrained by attention-head divisibility, e.g.
#                                 for GPT-OSS-120B use VLLM_TP=1 VLLM_PP=3)
#   VLLM_GPU_MEMORY_UTILIZATION   default: 0.9
#   VLLM_MAX_MODEL_LEN            default: 8192 (small for short prompts; raise as needed)
#   VLLM_CPU_OFFLOAD_GB           default: 0 (optional CPU offload budget)
#   VLLM_DTYPE                    default: auto
#   VLLM_QUANTIZATION             default: (auto-detect from config.json)
#   VLLM_EXTRA_ARGS               extra flags appended verbatim to `vllm serve`
#   WAIT_TIMEOUT_S                default: 1800 (large models can take a while to warm up)

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <model_weights> [served_model_name]" >&2
  exit 2
fi

MODEL_WEIGHTS="$1"
SERVED_NAME="${2:-$MODEL_WEIGHTS}"

VLLM_PYTHON="${VLLM_PYTHON:-$HOME/git/vllm/.venv/bin/python}"
VLLM_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES:-0,1,2}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
VLLM_PORT="${VLLM_PORT:-18000}"
VLLM_TP="${VLLM_TP:-3}"
VLLM_PP="${VLLM_PP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-1800}"
LOG_DIR="/tmp/vllm-tp"

OLLAMA_PORTS=(11435 11436 11437)

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Preconditions ---
for port in "${OLLAMA_PORTS[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"; then
    die "Ollama shard on port ${port} is still up. Run ollama-shards-down.sh first."
  fi
done

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${VLLM_PORT}$"; then
  die "Port ${VLLM_PORT} is already in use. Run vllm-tp-down.sh first."
fi

[ -x "$VLLM_PYTHON" ] || die "VLLM_PYTHON=$VLLM_PYTHON does not exist or is not executable"
"$VLLM_PYTHON" -c "import vllm" 2>/dev/null || die "vllm not importable from $VLLM_PYTHON"

mkdir -p "$LOG_DIR"

extra_flags=()
if [ -n "$VLLM_API_KEY" ]; then
  extra_flags+=(--api-key "$VLLM_API_KEY")
fi
if [ -n "$VLLM_QUANTIZATION" ]; then
  extra_flags+=(--quantization "$VLLM_QUANTIZATION")
fi
# shellcheck disable=SC2206
extra_verbatim=($VLLM_EXTRA_ARGS)

log_file="$LOG_DIR/vllm-${VLLM_PORT}.log"
log "Starting vLLM TP=${VLLM_TP} PP=${VLLM_PP} on port ${VLLM_PORT}"
log "  visible_devices=${VLLM_VISIBLE_DEVICES}"
log "  model=${MODEL_WEIGHTS}"
log "  served_as=${SERVED_NAME}"
log "  max_model_len=${VLLM_MAX_MODEL_LEN}"
log "  cpu_offload_gb=${VLLM_CPU_OFFLOAD_GB}"
log "  log=${log_file}"

CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE_DEVICES" \
  nohup "$VLLM_PYTHON" -c 'from vllm.entrypoints.cli.main import main; main()' serve "$MODEL_WEIGHTS" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --pipeline-parallel-size "$VLLM_PP" \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB" \
    --dtype "$VLLM_DTYPE" \
    "${extra_flags[@]}" \
    "${extra_verbatim[@]}" \
    >"$log_file" 2>&1 </dev/null &

# --- Wait for /v1/models ---
auth_header=()
if [ -n "$VLLM_API_KEY" ]; then
  auth_header=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

waited=0
while true; do
  if curl -fsS -o /dev/null "${auth_header[@]}" "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null; then
    log "vLLM is answering /v1/models on port ${VLLM_PORT}"
    break
  fi
  sleep 5
  waited=$((waited + 5))
  if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
    die "vLLM failed to answer /v1/models within ${WAIT_TIMEOUT_S}s. See ${log_file}"
  fi
  if [ $((waited % 30)) -eq 0 ]; then
    log "Still waiting for vLLM (${waited}s elapsed)"
  fi
done

log "vLLM TP=${VLLM_TP} is up: http://0.0.0.0:${VLLM_PORT} (served as ${SERVED_NAME})"
