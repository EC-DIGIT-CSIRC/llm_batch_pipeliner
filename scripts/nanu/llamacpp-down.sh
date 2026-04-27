#!/usr/bin/env bash
# Stop the native llama.cpp server started by llamacpp-up.sh.
#
# Environment:
#   LLAMACPP_PORT      default: 18100
#   STOP_TIMEOUT_S     default: 30

set -euo pipefail

LLAMACPP_PORT="${LLAMACPP_PORT:-18100}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-30}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

pids="$(sudo lsof -t -nP -iTCP:"$LLAMACPP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -z "$pids" ]; then
  log "Port ${LLAMACPP_PORT}: no listener"
else
  log "Port ${LLAMACPP_PORT}: sending TERM to PIDs: $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r sudo kill -TERM
fi

sudo pkill -TERM -f 'llama-server' 2>/dev/null || true

waited=0
while true; do
  still="$(sudo lsof -t -nP -iTCP:"$LLAMACPP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$still" ]; then
    log "Port ${LLAMACPP_PORT}: stopped"
    break
  fi
  sleep 1
  waited=$((waited + 1))
  if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then
    log "Port ${LLAMACPP_PORT}: still alive after ${STOP_TIMEOUT_S}s, sending KILL"
    echo "$still" | xargs -r sudo kill -KILL || true
    sudo pkill -KILL -f 'llama-server' 2>/dev/null || true
    break
  fi
done

if ss -ltn | awk '{print $4}' | grep -qE "[:.]${LLAMACPP_PORT}$"; then
  log "ERROR: Port ${LLAMACPP_PORT} is still bound after shutdown"
  exit 1
fi

log "llama.cpp server stopped"
