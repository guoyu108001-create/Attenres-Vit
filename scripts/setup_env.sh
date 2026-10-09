#!/usr/bin/env bash
# One-time environment setup on the training server: ./.venv with torch 2.10.0+cu128,
# vendored timm and attnres_vit (editable). Activate later with: source .venv/bin/activate
#
# Why not conda / plain pip (147, 2026-10): conda.anaconda.org is unreachable and the defaults
# channel needs ToS acceptance; the Aliyun mirrors throttle HTTP/1.1 (pip, uv) to ~0.24 MB/s but
# serve HTTP/2 (curl) at ~25 MB/s. So torch, torchvision and torch's pinned nvidia/triton wheels
# are fetched with curl, and uv installs from those files (only small packages come from the index).
#
# Usage:  bash scripts/setup_env.sh
# Env:    PYTHON     CPython 3.13 to build the venv from (default ~/miniconda3/bin/python3)
#         PIP_INDEX  PEP 503 simple index (default https://mirrors.aliyun.com/pypi/simple/)
#         WHEELS     PyTorch wheel mirror (default https://mirrors.aliyun.com/pytorch-wheels/cu128)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"
export PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
WHEELS="${WHEELS:-https://mirrors.aliyun.com/pytorch-wheels/cu128}"
TORCH=torch-2.10.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl
TORCHVISION=torchvision-0.25.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl
UV=uv-0.9.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
UV_URL="${PIP_INDEX%/simple/}/packages/4c/4a/d5357825bb47ff73762d247b1a553a966fef6802e3ab829fe60934cbf339/$UV"

echo "==> venv ($("$PYTHON" --version)) in $PROJECT_ROOT/.venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> curl torch / torchvision / uv wheels into .wheels/"
mkdir -p .wheels
curl -fsSL --retry 3 -o ".wheels/$TORCH" "$WHEELS/$TORCH"
curl -fsSL --retry 3 -o ".wheels/$TORCHVISION" "$WHEELS/$TORCHVISION"
curl -fsSL --retry 3 -o ".wheels/$UV" "$UV_URL"
pip install --no-index ".wheels/$UV"

echo "==> curl torch's pinned Linux dependency wheels"
python scripts/lib/fetch_torch_deps.py ".wheels/$TORCH" .wheels

echo "==> uv install: local wheels + small deps, vendored timm, attnres_vit"
uv pip install --python .venv/bin/python -i "$PIP_INDEX" .wheels/*.whl -e repos/timm -e .
python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), '| timm', timm.__version__, timm.__file__)"

echo "==> smoke test (forward pass of all models + AttnRes == baseline at init)"
python tests/test_forward.py

echo ""
echo "Setup complete. Next: set data_dir in configs/phase0/*.yaml (RUN.md §3), then on each server:"
echo "  source .venv/bin/activate && NODE=a bash scripts/run_phase0_node.sh   # NODE=b on the other server"
