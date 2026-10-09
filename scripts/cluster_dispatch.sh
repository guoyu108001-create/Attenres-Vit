#!/usr/bin/env bash
# Optional: start the Phase-0 tmux queues on both servers via SSH.
#
# Edit SERVER_A / SERVER_B / REMOTE_PROJECT / REMOTE_ENV before use.
# Requires passwordless SSH and the project at the same path on both nodes.
# REMOTE_ENV must put the project's python / torchrun on PATH in a non-interactive SSH shell.
#
# Usage:
#   bash scripts/cluster_dispatch.sh

set -euo pipefail

# ---------- EDIT THESE ----------
SERVER_A="${SERVER_A:-user@server-a}"
SERVER_B="${SERVER_B:-user@server-b}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/path/to/attnres}"
REMOTE_ENV="${REMOTE_ENV:-source $REMOTE_PROJECT/.venv/bin/activate}"
# --------------------------------

ssh "$SERVER_A" "$REMOTE_ENV && cd $REMOTE_PROJECT && NODE=a bash scripts/run_phase0_node.sh"
ssh "$SERVER_B" "$REMOTE_ENV && cd $REMOTE_PROJECT && NODE=b bash scripts/run_phase0_node.sh"
echo ">>> Queues started. Check: ssh $SERVER_A tmux ls ; ssh $SERVER_B tmux ls"
