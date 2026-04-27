#!/usr/bin/env bash
# Spin up one temporary Ollama server on a single endpoint, restricted to a
# chosen GPU set (default: GPUs 0 and 1). This is the closest Ollama analogue
# to a tensor-parallel / no-sharding benchmark: one HTTP endpoint, one model,
# multiple GPUs available to the engine.
#
# Usage:
#   ollama-tp-up.sh [model_name]
#
# Example:
#   OLLAMA_VISIBLE_DEVICES=0,1 OLLAMA_PORT=11440 ollama-tp-up.sh qwen3.6:latest
#
# Preconditions:
#   - The system `ollama.service` must be STOPPED first to free the GPUs.
#   - No listener should already exist on OLLAMA_PORT.
#   - OLLAMA_NOPRUNE=true is enforced for safety.
#
# Environment:
#   OLLAMA_VISIBLE_DEVICES  default: 0,1
#   OLLAMA_PORT             default: 11440
#   OLLAMA_MODELS_DIR       default: /data/models/ollama
#   OLLAMA_NUM_PARALLEL     default: 1
#   WAIT_TIMEOUT_S          default: 180

set -euo pipefail

MODEL_NAME="${1:-qwen3.6:latest}"
OLLAMA_VISIBLE_DEVICES="${OLLAMA_VISIBLE_DEVICES:-0,1}"
OLLAMA_PORT="${OLLAMA_PORT:-11440}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/data/models/ollama}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-180}"
LOG_DIR="/tmp/ollama-tp"
OLLAMA_BIN="${OLLAMA_BIN:-/usr/local/bin/ollama}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

if systemctl is-active --quiet ollama; then
  die "system ollama.service is still running on :11434; stop it first to free the GPUs"
fi

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${OLLAMA_PORT}$"; then
  die "Port ${OLLAMA_PORT} is already in use. Run ollama-tp-down.sh first."
fi

[ -x "$OLLAMA_BIN" ] || die "Ollama binary not found at $OLLAMA_BIN"
[ -d "$OLLAMA_MODELS_DIR" ] || die "OLLAMA_MODELS_DIR=$OLLAMA_MODELS_DIR does not exist"

sudo mkdir -p "$LOG_DIR"
sudo chown ollama:ollama "$LOG_DIR"
log_file="$LOG_DIR/ollama-${OLLAMA_PORT}.log"

log "Starting temporary Ollama TP-style server on :${OLLAMA_PORT}"
log "  visible_devices=${OLLAMA_VISIBLE_DEVICES}"
log "  model=${MODEL_NAME}"
log "  log=${log_file}"

sudo -u ollama sh -lc "HOME=/usr/share/ollama PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES='$OLLAMA_VISIBLE_DEVICES' OLLAMA_HOST='0.0.0.0:${OLLAMA_PORT}' OLLAMA_MODELS='$OLLAMA_MODELS_DIR' OLLAMA_NUM_PARALLEL='$OLLAMA_NUM_PARALLEL' OLLAMA_NOPRUNE=true nohup '$OLLAMA_BIN' serve >'$log_file' 2>&1 </dev/null &"

waited=0
while true; do
  if curl -fsS -o /dev/null "http://127.0.0.1:${OLLAMA_PORT}/api/tags" 2>/dev/null; then
    log "Temporary Ollama server is answering /api/tags on :${OLLAMA_PORT}"
    break
  fi
  sleep 2
  waited=$((waited + 2))
  if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
    die "Temporary Ollama server failed to answer /api/tags within ${WAIT_TIMEOUT_S}s. See ${log_file}"
  fi
done

log "Warm-loading model ${MODEL_NAME}"
curl -fsS -X POST "http://127.0.0.1:${OLLAMA_PORT}/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
  >/dev/null || die "Model warmup failed for ${MODEL_NAME}. See ${log_file}"

log "Temporary Ollama server is ready on http://0.0.0.0:${OLLAMA_PORT}"
