#!/usr/bin/env bash
# Stop the 3 Ollama shard daemons on ports 11435/11436/11437.
# Does NOT touch the system ollama.service (port 11434).
#
# Environment:
#   STOP_TIMEOUT_S  default: 10

set -euo pipefail

STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-10}"
PORTS=(11435 11436 11437)

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

log "All Ollama shards stopped"
