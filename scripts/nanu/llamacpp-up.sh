#!/usr/bin/env bash
# Start a native llama.cpp llama-server on nanu.
#
# Environment:
#   LLAMACPP_BIN             default: /home/aaron/git/llama.cpp/build/bin/llama-server
#   LLAMACPP_VISIBLE_DEVICES default: 0,1,2
#   LLAMACPP_DEVICE_LIST     default: CUDA0..CUDA{n-1} for visible devices
#   LLAMACPP_PORT           default: 18100
#   LLAMACPP_SPLIT_MODE     default: layer
#   LLAMACPP_TENSOR_SPLIT   default: 1,1,1
#   LLAMACPP_N_GPU_LAYERS   default: all
#   LLAMACPP_CTX_SIZE       default: 32768
#   LLAMACPP_BATCH_SIZE     default: 2048
#   LLAMACPP_UBATCH_SIZE    default: 512
#   LLAMACPP_PARALLEL       default: 1
#   LLAMACPP_THREADS        default: 16
#   LLAMACPP_THREADS_BATCH  default: 16
#   LLAMACPP_THREADS_HTTP   default: 8
#   LLAMACPP_FLASH_ATTN     default: on
#   LLAMACPP_REASONING      default: off
#   LLAMACPP_CACHE_PROMPT   default: true
#   LLAMACPP_CONT_BATCHING  default: true
#   LLAMACPP_API_KEY        optional bearer token
#   LLAMACPP_EXTRA_ARGS     extra flags appended verbatim to llama-server
#   LLAMACPP_ENDPOINT       default: responses (responses|chat)
#   WAIT_TIMEOUT_S          default: 1800

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <model_path> [served_model_name]" >&2
  exit 2
fi

MODEL_PATH="$1"
SERVED_NAME="${2:-qwen3.6:latest}"

LLAMACPP_BIN="${LLAMACPP_BIN:-$HOME/git/llama.cpp/build/bin/llama-server}"
LLAMACPP_VISIBLE_DEVICES="${LLAMACPP_VISIBLE_DEVICES:-0,1,2}"
if [ -z "${LLAMACPP_DEVICE_LIST:-}" ]; then
  IFS=',' read -r -a visible_gpu_ids <<< "$LLAMACPP_VISIBLE_DEVICES"
  device_names=()
  for idx in "${!visible_gpu_ids[@]}"; do
    device_names+=("CUDA${idx}")
  done
  LLAMACPP_DEVICE_LIST="$(IFS=,; printf '%s' "${device_names[*]}")"
else
  LLAMACPP_DEVICE_LIST="$LLAMACPP_DEVICE_LIST"
fi
LLAMACPP_PORT="${LLAMACPP_PORT:-18100}"
LLAMACPP_SPLIT_MODE="${LLAMACPP_SPLIT_MODE:-layer}"
LLAMACPP_TENSOR_SPLIT="${LLAMACPP_TENSOR_SPLIT:-1,1,1}"
LLAMACPP_N_GPU_LAYERS="${LLAMACPP_N_GPU_LAYERS:-all}"
LLAMACPP_CTX_SIZE="${LLAMACPP_CTX_SIZE:-32768}"
LLAMACPP_BATCH_SIZE="${LLAMACPP_BATCH_SIZE:-2048}"
LLAMACPP_UBATCH_SIZE="${LLAMACPP_UBATCH_SIZE:-512}"
LLAMACPP_PARALLEL="${LLAMACPP_PARALLEL:-1}"
LLAMACPP_THREADS="${LLAMACPP_THREADS:-16}"
LLAMACPP_THREADS_BATCH="${LLAMACPP_THREADS_BATCH:-16}"
LLAMACPP_THREADS_HTTP="${LLAMACPP_THREADS_HTTP:-8}"
LLAMACPP_FLASH_ATTN="${LLAMACPP_FLASH_ATTN:-on}"
LLAMACPP_REASONING="${LLAMACPP_REASONING:-off}"
LLAMACPP_CACHE_PROMPT="${LLAMACPP_CACHE_PROMPT:-true}"
LLAMACPP_CONT_BATCHING="${LLAMACPP_CONT_BATCHING:-true}"
LLAMACPP_API_KEY="${LLAMACPP_API_KEY:-}"
LLAMACPP_EXTRA_ARGS="${LLAMACPP_EXTRA_ARGS:-}"
LLAMACPP_ENDPOINT="${LLAMACPP_ENDPOINT:-responses}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-1800}"
LOG_DIR="/tmp/llamacpp"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

[ -f "$MODEL_PATH" ] || die "Model not found: $MODEL_PATH"
[ -x "$LLAMACPP_BIN" ] || die "llama-server not found or not executable: $LLAMACPP_BIN"

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${LLAMACPP_PORT}$"; then
  die "Port ${LLAMACPP_PORT} is already in use. Run llamacpp-down.sh first."
fi

mkdir -p "$LOG_DIR"

extra_flags=()
if [ -n "$LLAMACPP_API_KEY" ]; then
  extra_flags+=(--api-key "$LLAMACPP_API_KEY")
fi
if [ "$LLAMACPP_CACHE_PROMPT" = "true" ]; then
  extra_flags+=(--cache-prompt)
else
  extra_flags+=(--no-cache-prompt)
fi
if [ "$LLAMACPP_CONT_BATCHING" = "true" ]; then
  extra_flags+=(--cont-batching)
else
  extra_flags+=(--no-cont-batching)
fi
# shellcheck disable=SC2206
extra_verbatim=($LLAMACPP_EXTRA_ARGS)

log_file="$LOG_DIR/llamacpp-${LLAMACPP_PORT}.log"
log "Starting llama.cpp on port ${LLAMACPP_PORT}"
log "  visible_devices=${LLAMACPP_VISIBLE_DEVICES}"
log "  device_list=${LLAMACPP_DEVICE_LIST}"
log "  model=${MODEL_PATH}"
log "  served_as=${SERVED_NAME}"
log "  endpoint=${LLAMACPP_ENDPOINT}"
log "  log=${log_file}"

CUDA_VISIBLE_DEVICES="$LLAMACPP_VISIBLE_DEVICES" \
  nohup "$LLAMACPP_BIN" \
    --host 0.0.0.0 \
    --port "$LLAMACPP_PORT" \
    -m "$MODEL_PATH" \
    -a "$SERVED_NAME" \
    --device "$LLAMACPP_DEVICE_LIST" \
    --split-mode "$LLAMACPP_SPLIT_MODE" \
    --tensor-split "$LLAMACPP_TENSOR_SPLIT" \
    --n-gpu-layers "$LLAMACPP_N_GPU_LAYERS" \
    --main-gpu 0 \
    --ctx-size "$LLAMACPP_CTX_SIZE" \
    --batch-size "$LLAMACPP_BATCH_SIZE" \
    --ubatch-size "$LLAMACPP_UBATCH_SIZE" \
    --parallel "$LLAMACPP_PARALLEL" \
    --threads "$LLAMACPP_THREADS" \
    --threads-batch "$LLAMACPP_THREADS_BATCH" \
    --threads-http "$LLAMACPP_THREADS_HTTP" \
    --flash-attn "$LLAMACPP_FLASH_ATTN" \
    --reasoning "$LLAMACPP_REASONING" \
    --timeout 1200 \
    --metrics \
    "${extra_flags[@]}" \
    "${extra_verbatim[@]}" \
    >"$log_file" 2>&1 </dev/null &

auth_header=()
if [ -n "$LLAMACPP_API_KEY" ]; then
  auth_header=(-H "Authorization: Bearer $LLAMACPP_API_KEY")
fi

waited=0
while true; do
  if curl -fsS -o /dev/null "${auth_header[@]}" "http://127.0.0.1:${LLAMACPP_PORT}/v1/models" 2>/dev/null; then
    log "llama.cpp is answering /v1/models on port ${LLAMACPP_PORT}"
    break
  fi
  sleep 5
  waited=$((waited + 5))
  if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
    die "llama.cpp failed to answer /v1/models within ${WAIT_TIMEOUT_S}s. See ${log_file}"
  fi
  if [ $((waited % 30)) -eq 0 ]; then
    log "Still waiting for llama.cpp (${waited}s elapsed)"
  fi
done

warmup_path="/v1/responses"
warmup_body='{"model":"qwen3.6:latest","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"stream":false}'
if [ "$LLAMACPP_ENDPOINT" = "chat" ]; then
  warmup_path="/v1/chat/completions"
  warmup_body='{"model":"qwen3.6:latest","messages":[{"role":"user","content":"hi"}],"stream":false}'
fi

log "Warm-loading model ${SERVED_NAME}"
curl -fsS -X POST "http://127.0.0.1:${LLAMACPP_PORT}${warmup_path}" \
  -H 'Content-Type: application/json' \
  "${auth_header[@]}" \
  -d "$warmup_body" \
  >/dev/null || die "Model warmup failed for ${SERVED_NAME}. See ${log_file}"

log "llama.cpp server is ready on http://0.0.0.0:${LLAMACPP_PORT}"
