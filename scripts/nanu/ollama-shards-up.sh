#!/usr/bin/env bash
# Spin up 3 Ollama shard daemons on ports 11435/11436/11437, pinned to GPU 0/1/2.
#
# Preconditions:
#   - System ollama.service on 11434 is untouched (keeps coexisting).
#   - No Ollama shard daemon already listening on 11435/11436/11437.
#   - OLLAMA_NOPRUNE=true is forced on every shard (see AGENTS.md safety rule).
#
# Environment:
#   OLLAMA_MODELS_DIR    default: /data/models/ollama
#   OLLAMA_NUM_PARALLEL  default: 1
#   WAIT_TIMEOUT_S       default: 120
#
# Exits non-zero if any shard fails to start or answer /api/tags.

set -euo pipefail

OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/data/models/ollama}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-120}"
LOG_DIR="/tmp/ollama-shards"
OLLAMA_BIN="${OLLAMA_BIN:-/usr/local/bin/ollama}"

SHARDS=(
  "11435:0"
  "11436:1"
  "11437:2"
)

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Preconditions ---
for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"; then
    die "Port ${port} is already in use. Run ollama-shards-down.sh first."
  fi
done

[ -x "$OLLAMA_BIN" ] || die "Ollama binary not found or not executable at $OLLAMA_BIN"
[ -d "$OLLAMA_MODELS_DIR" ] || die "OLLAMA_MODELS_DIR=$OLLAMA_MODELS_DIR does not exist"

sudo mkdir -p "$LOG_DIR"
sudo chown ollama:ollama "$LOG_DIR"

# --- Start shards ---
for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  gpu="${spec##*:}"
  log_file="$LOG_DIR/ollama-${port}.log"
  log "Starting Ollama shard port=${port} gpu=${gpu} log=${log_file}"

  sudo -u ollama env \
    HOME=/usr/share/ollama \
    PATH=/usr/local/bin:/usr/bin:/bin \
    CUDA_VISIBLE_DEVICES="$gpu" \
    OLLAMA_HOST="0.0.0.0:${port}" \
    OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
    OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" \
    OLLAMA_NOPRUNE=true \
    nohup "$OLLAMA_BIN" serve >"$log_file" 2>&1 </dev/null &
done

# --- Wait for /api/tags on each shard ---
for spec in "${SHARDS[@]}"; do
  port="${spec%%:*}"
  waited=0
  while true; do
    if curl -fsS -o /dev/null "http://127.0.0.1:${port}/api/tags" 2>/dev/null; then
      log "Shard ${port} is answering /api/tags"
      break
    fi
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
      die "Shard ${port} failed to answer /api/tags within ${WAIT_TIMEOUT_S}s. See $LOG_DIR/ollama-${port}.log"
    fi
  done
done

log "All 3 Ollama shards are up: 11435 11436 11437"
