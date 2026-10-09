#!/usr/bin/env python3
"""ImageNet training entry point.

Registers AttnRes ViT models with timm, then delegates to timm's train.py.
Optionally enables stricter reproducibility via ATTNRES_DETERMINISTIC=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMM_ROOT = ROOT / "repos" / "timm"

if TIMM_ROOT.is_dir():
    sys.path.insert(0, str(TIMM_ROOT))
sys.path.insert(0, str(ROOT))

import attnres_vit  # noqa: F401  — side-effect: register models with timm


def _apply_deterministic_mode() -> None:
    """Best-effort deterministic training (slower, may not be bit-exact on all GPUs)."""
    import torch

    if os.environ.get("ATTNRES_DETERMINISTIC", "0") != "1":
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("[attnres] Deterministic mode enabled (ATTNRES_DETERMINISTIC=1)")


if __name__ == "__main__":
    _apply_deterministic_mode()
    from train import main

    main()
