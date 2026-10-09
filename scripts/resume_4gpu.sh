#!/usr/bin/env bash
# Resume a run from its best epoch inside tmux, writing into the same run directory.
#
# Usage:
#   bash scripts/resume_4gpu.sh <config.yaml> <run_dir | checkpoint.pth.tar> [gpu_ids] [master_port]
#
# Example:
#   bash scripts/resume_4gpu.sh configs/phase0/e01_d12_baseline.yaml \
#     output/phase0/20261008_153045_e01_d12_baseline_s42 0,1,2,3 29500
#
# A run directory resolves to model_best.pth.tar (best eval epoch, EMA metric); training
# continues from the epoch after it. Pass last.pth.tar explicitly to continue from the latest epoch.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.yaml> <run_dir | checkpoint.pth.tar> [gpu_ids] [master_port]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/lib/run_utils.sh"

CONFIG="$1"
CKPT="$2"
GPUS="${3:-0,1,2,3}"
PORT="${4:-29500}"

if [[ -d "$CKPT" ]]; then
  CKPT="$CKPT/model_best.pth.tar"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT"
  exit 1
fi

# Absolute paths: the tmux session starts in PROJECT_ROOT.
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

tmux_launch "resume_$(basename "$(dirname "$CKPT")")" \
  env RESUME="$CKPT" bash scripts/run_queue.sh "$GPUS" "$PORT" "$CONFIG"
