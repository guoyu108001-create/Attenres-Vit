#!/usr/bin/env bash
# Train one experiment on 4 GPUs with reproducibility + anti-overwrite.
#
# Usage:
#   bash scripts/train_4gpu.sh <config.yaml> [gpu_ids] [extra args...]
#
# Environment variables:
#   SEED=42              Random seed (default 42, passed to timm --seed)
#   RUN_ID=20261008_120000  Optional fixed run timestamp (default: now)
#   FORCE=1              Allow reusing an existing output directory
#   RESUME=/path/to.ckpt Resume training (skips anti-overwrite check)
#   ATTNRES_DETERMINISTIC=1  Enable stricter (slower) deterministic mode
#
# Output directory pattern:
#   output/phase0/{RUN_ID}_{experiment_slug}_s{SEED}/
#   Example: output/phase0/20261008_153045_e01_d12_baseline_s42/

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [gpu_ids] [extra train args...]"
  exit 1
fi

CONFIG="$1"
shift

GPUS="0,1,2,3"
if [[ $# -gt 0 && "$1" != --* ]]; then
  GPUS="$1"
  shift
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/lib/run_utils.sh"

cd "$PROJECT_ROOT"

SEED="${SEED:-$(yaml_get seed "$CONFIG")}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

EXP_SLUG="$(yaml_get experiment "$CONFIG")"
if [[ -z "$EXP_SLUG" ]]; then
  EXP_SLUG="$(basename "$CONFIG" .yaml)"
fi

MODEL="$(yaml_get model "$CONFIG")"
OUTPUT_PARENT="$(yaml_get output "$CONFIG")"
OUTPUT_PARENT="${OUTPUT_PARENT:-./output/phase0}"

EXTRA_ARGS=("$@")
if [[ -n "${RESUME:-}" ]]; then
  # Continue in the same run directory as the checkpoint.
  OUTDIR="$(cd "$(dirname "$RESUME")" && pwd)"
  RUN_NAME="$(basename "$OUTDIR")"
  OUTPUT_PARENT="$(dirname "$OUTDIR")"
  EXTRA_ARGS=(--resume "$RESUME" "${EXTRA_ARGS[@]}")
  echo "[INFO] Resume mode: reusing $OUTDIR"
else
  RUN_NAME="$(build_run_name "$EXP_SLUG" "$SEED" "$RUN_ID")"
  OUTDIR="$(python - <<PY
import os
print(os.path.normpath(os.path.join("${OUTPUT_PARENT}", "${RUN_NAME}")))
PY
)"
  assert_output_safe "$OUTDIR"
  write_run_manifest "$OUTDIR" "$CONFIG" "$RUN_NAME" "$SEED" "$GPUS" "${EXTRA_ARGS[@]}"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SEED
export RUN_ID
export RUN_NAME
export OUTDIR

NUM_GPUS="$(echo "$GPUS" | tr ',' '\n' | wc -l)"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo " Config     : $CONFIG"
echo " Model      : $MODEL"
echo " Run name   : $RUN_NAME"
echo " Output dir : $OUTDIR"
echo " Seed       : $SEED"
echo " GPUs       : $GPUS ($NUM_GPUS processes)"
echo " Port       : $MASTER_PORT"
echo " Resume     : ${RESUME:-none}"
echo " Extra args : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train_imagenet.py" \
  -c "$CONFIG" \
  --seed "$SEED" \
  --output "$OUTPUT_PARENT" \
  --experiment "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"
