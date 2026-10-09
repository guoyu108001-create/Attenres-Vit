#!/usr/bin/env python3
"""ImageNet validation entry point with AttnRes ViT models registered."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMM_ROOT = ROOT / "repos" / "timm"

if TIMM_ROOT.is_dir():
    sys.path.insert(0, str(TIMM_ROOT))
sys.path.insert(0, str(ROOT))

import attnres_vit  # noqa: F401

from validate import main


if __name__ == "__main__":
    main()
