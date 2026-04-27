#!/usr/bin/env bash
# Stop the 3 vLLM shard servers on ports 18000/18001/18002.
#
# Environment:
#   STOP_TIMEOUT_S  default: 20 (vLLM takes a bit to release GPU memory)

set -euo pipefail

STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-20}"
PORTS=(18000 18001 18002)

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

for port in "${PORTS[@]}"; do
  pids="$(sudo lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    log "Port ${port}: no listener"
    continue
  fi

  log "Port ${port}: sending TERM to PIDs: $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r sudo kill -TERM

  waited=0
  while true; do
    still="$(sudo lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -z "$still" ]; then
      log "Port ${port}: stopped cleanly"
      break
    fi
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then
      log "Port ${port}: still alive after ${STOP_TIMEOUT_S}s, sending KILL"
      echo "$still" | xargs -r sudo kill -KILL || true
      break
    fi
  done
done

# Final verification
for port in "${PORTS[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"; then
    log "ERROR: Port ${port} is still bound after shutdown"
    exit 1
  fi
done

log "All vLLM shards stopped"
