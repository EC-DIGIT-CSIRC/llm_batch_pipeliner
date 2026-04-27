#!/usr/bin/env bash
# Verify that GPUs 0/1/2 have <500 MiB used (i.e. no leftover model from
# a previous engine). Use this between sequential benchmarks (Ollama -> vLLM).
#
# Environment:
#   MAX_USED_MIB     default: 500
#   GPUS             default: "0,1,2"
#   MAX_WAIT_S       default: 60   (wait this long for GPUs to clear)

set -euo pipefail

MAX_USED_MIB="${MAX_USED_MIB:-500}"
GPUS="${GPUS:-0,1,2}"
MAX_WAIT_S="${MAX_WAIT_S:-60}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

waited=0
while true; do
  busy=()
  while IFS=, read -r idx used; do
    idx="$(echo "$idx" | tr -d ' ')"
    used="$(echo "$used" | tr -d ' MiB')"
    if [ "$used" -gt "$MAX_USED_MIB" ]; then
      busy+=("gpu${idx}=${used}MiB")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS")

  if [ "${#busy[@]}" -eq 0 ]; then
    log "All GPUs (${GPUS}) idle: <${MAX_USED_MIB} MiB used"
    exit 0
  fi

  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    die "GPUs still busy after ${MAX_WAIT_S}s: ${busy[*]}"
  fi

  log "Waiting for GPUs to clear: ${busy[*]} (waited ${waited}s)"
  sleep 5
  waited=$((waited + 5))
done
