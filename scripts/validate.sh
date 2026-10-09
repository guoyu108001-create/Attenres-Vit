#!/usr/bin/env bash
# Evaluate a trained checkpoint on ImageNet val set (EMA weights).
#
# Training runs with model_ema, so timm logs EMA accuracy to summary.csv and picks
# model_best.pth.tar by it; --use-ema evaluates the same weights.
#
# Usage:
#   bash scripts/validate.sh <config.yaml> <checkpoint.pth> [gpu_id]

set -euo pipefail

CONFIG="$1"
CHECKPOINT="$2"
GPU="${3:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Extract model name from yaml config
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model'])")
DATA_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_dir'])")

python "$PROJECT_ROOT/validate_imagenet.py" \
  "$DATA_DIR" \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --batch-size 256 \
  --use-ema \
  --amp
