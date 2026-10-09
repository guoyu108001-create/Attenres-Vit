#!/usr/bin/env bash
# Launch Phase-0 on ONE 8-GPU server as two tmux queues (GPU 0-3 / GPU 4-7).
#
# Cluster layout (2 servers x 8 A40). Each queue runs its jobs back-to-back:
#   NODE=a  GPU 0-3: E01 -> E07     GPU 4-7: E02 -> E08
#   NODE=b  GPU 0-3: E03 -> E05     GPU 4-7: E04 -> E06
#
# Baseline/AttnRes pairs still run side by side on the same server. The two longest
# jobs (d48 wide, ~13 d) follow the two shortest (d12, ~4 d), so both servers finish
# in ~17 d / ~13.5 d instead of ~11.5 d / ~19 d, and no GPU waits between jobs.
#
# Usage (on each server, with the project's venv active: source .venv/bin/activate):
#   NODE=a bash scripts/run_phase0_node.sh
#   NODE=b bash scripts/run_phase0_node.sh
#
# Monitor:
#   tmux ls
#   tmux attach -t p0_a_g0                      # detach: Ctrl-b d
#   tail -f logs/phase0/e01_d12_baseline.log
# Stop a whole queue:
#   tmux kill-session -t p0_a_g0
#
# Environment:
#   NODE=a|b          Required. Which server this is.
#   GPUS_G0 / GPUS_G1 GPU ids of the two queues (default 0,1,2,3 / 4,5,6,7), e.g. GPUS_G0=0,5,6,7
#                     GPUS_G1=1,2,3,4 when another user holds GPU 1-4: queue g1 then waits for them.
#   MASTER_PORT_BASE  torchrun port of queue g0 (default 29500); queue g1 uses BASE+1.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/lib/run_utils.sh"

NODE="${NODE:-}"
BASE="${MASTER_PORT_BASE:-29500}"
C=configs/phase0

case "$NODE" in
  a)
    Q0=("$C/e01_d12_baseline.yaml" "$C/e07_d48_wide_baseline.yaml")
    Q1=("$C/e02_d12_attnres.yaml" "$C/e08_d48_wide_attnres.yaml")
    ;;
  b)
    Q0=("$C/e03_d24_baseline.yaml" "$C/e05_d48_narrow_baseline.yaml")
    Q1=("$C/e04_d24_attnres.yaml" "$C/e06_d48_narrow_attnres.yaml")
    ;;
  *)
    echo "[ERROR] Set NODE=a or NODE=b to identify this server."
    echo "  Example: NODE=a bash scripts/run_phase0_node.sh"
    exit 1
    ;;
esac

tmux_launch "p0_${NODE}_g0" bash scripts/run_queue.sh "${GPUS_G0:-0,1,2,3}" "$BASE" "${Q0[@]}"
tmux_launch "p0_${NODE}_g1" bash scripts/run_queue.sh "${GPUS_G1:-4,5,6,7}" "$((BASE + 1))" "${Q1[@]}"
