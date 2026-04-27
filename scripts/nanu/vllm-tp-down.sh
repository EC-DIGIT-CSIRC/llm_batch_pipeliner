#!/usr/bin/env bash
# Stop the single tensor-parallel vLLM server (default port 18000).
#
# Environment:
#   VLLM_PORT       default: 18000
#   STOP_TIMEOUT_S  default: 30 (vLLM TP needs more time to release GPU memory)

set -euo pipefail

VLLM_PORT="${VLLM_PORT:-18000}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-30}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

pids="$(sudo lsof -t -nP -iTCP:"$VLLM_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -z "$pids" ]; then
  log "Port ${VLLM_PORT}: no listener"
else
  log "Port ${VLLM_PORT}: sending TERM to PIDs: $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r sudo kill -TERM
fi

# Also kill any orphan vllm/python workers that may have been spawned for TP
sudo pkill -TERM -f 'vllm.entrypoints.cli.main.*serve' 2>/dev/null || true

waited=0
while true; do
  still="$(sudo lsof -t -nP -iTCP:"$VLLM_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$still" ]; then
    log "Port ${VLLM_PORT}: stopped"
    break
  fi
  sleep 1
  waited=$((waited + 1))
  if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then
    log "Port ${VLLM_PORT}: still alive after ${STOP_TIMEOUT_S}s, sending KILL"
    echo "$still" | xargs -r sudo kill -KILL || true
    sudo pkill -KILL -f 'vllm.entrypoints.cli.main.*serve' 2>/dev/null || true
    break
  fi
done

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${VLLM_PORT}$"; then
  log "ERROR: Port ${VLLM_PORT} is still bound after shutdown"
  exit 1
fi

log "vLLM TP server stopped"
