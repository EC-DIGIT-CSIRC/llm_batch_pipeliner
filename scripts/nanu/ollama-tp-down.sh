#!/usr/bin/env bash
# Stop the temporary single-endpoint Ollama server started by ollama-tp-up.sh.
# Does not touch the system `ollama.service`.
#
# Environment:
#   OLLAMA_PORT      default: 11440
#   STOP_TIMEOUT_S   default: 20

set -euo pipefail

OLLAMA_PORT="${OLLAMA_PORT:-11440}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-20}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

pids="$(sudo lsof -t -nP -iTCP:"$OLLAMA_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -z "$pids" ]; then
  log "Port ${OLLAMA_PORT}: no listener"
else
  log "Port ${OLLAMA_PORT}: sending TERM to PIDs: $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r sudo kill -TERM
fi

waited=0
while true; do
  still="$(sudo lsof -t -nP -iTCP:"$OLLAMA_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$still" ]; then
    log "Port ${OLLAMA_PORT}: stopped cleanly"
    break
  fi
  sleep 1
  waited=$((waited + 1))
  if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then
    log "Port ${OLLAMA_PORT}: still alive after ${STOP_TIMEOUT_S}s, sending KILL"
    echo "$still" | xargs -r sudo kill -KILL || true
    break
  fi
done

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${OLLAMA_PORT}$"; then
  log "ERROR: Port ${OLLAMA_PORT} is still bound after shutdown"
  exit 1
fi

log "Temporary Ollama server stopped"
