#!/usr/bin/env bash
# Run configs back-to-back on one 4-GPU group. Meant to live inside tmux
# (started by run_phase0_node.sh / resume_4gpu.sh).
#
# Before each job it waits until the GPUs are free; a failed job does not stop the queue.
#
# Usage:
#   bash scripts/run_queue.sh <gpu_ids> <master_port> <config.yaml> [config.yaml ...]
#
# Log: logs/phase0/<config name>.log (appended)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/lib/run_utils.sh"
set +e  # keep the queue going after a failed job

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu_ids> <master_port> <config.yaml> [config.yaml ...]"
  exit 1
fi

GPUS="$1"
PORT="$2"
shift 2

LOG_DIR="$PROJECT_ROOT/logs/phase0"
mkdir -p "$LOG_DIR"

for cfg in "$@"; do
  log="$LOG_DIR/$(basename "$cfg" .yaml).log"
  wait_gpus_free "$GPUS"
  echo "[$(date '+%F %T')] START $cfg  GPU=$GPUS  port=$PORT" | tee -a "$log"
  MASTER_PORT="$PORT" bash "$PROJECT_ROOT/scripts/train_4gpu.sh" "$cfg" "$GPUS" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   $cfg  exit=$rc" | tee -a "$log"
done
