#!/usr/bin/env python3
"""Smoke test: forward pass + parameter count for all registered models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "timm"))

import torch
import attnres_vit  # noqa: F401
from timm.models import create_model

MODELS = [
    "attnres_vit_tiny",
    "attnres_vit_tiny_attnres",
    "attnres_vit_d12_base",
    "attnres_vit_d12_base_attnres",
    "attnres_vit_d24_base",
    "attnres_vit_d24_base_attnres",
    "attnres_vit_d48_narrow",
    "attnres_vit_d48_narrow_attnres",
    "attnres_vit_d48_wide",
    "attnres_vit_d48_wide_attnres",
]


def check_init_equivalence(device: str) -> None:
    """Zero pseudo-queries => an AttnRes model equals the baseline with the same layer weights."""
    x = torch.randn(2, 3, 224, 224, device=device)
    for base_name in ["attnres_vit_tiny", "attnres_vit_d12_base"]:
        attn = create_model(base_name + "_attnres").to(device).eval()
        base = create_model(base_name).to(device).eval()
        missing, unexpected = base.load_state_dict(attn.state_dict(), strict=False)
        assert not missing and all(k.startswith("attn_res.") for k in unexpected)
        with torch.inference_mode():
            y_attn, y_base = attn.forward_features(x), base.forward_features(x)
        rel = ((y_attn - y_base).norm() / y_base.norm()).item()
        assert rel < 1e-3, f"{base_name}_attnres differs from {base_name} at init (rel {rel:.2e})"
        print(f"OK  {base_name + '_attnres':35s}  == {base_name} at init (rel diff {rel:.1e})")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 224, 224, device=device)

    print(f"Device: {device}")
    print("-" * 72)
    for name in MODELS:
        model = create_model(name, num_classes=1000).to(device)
        model.eval()
        with torch.inference_mode():
            y = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        assert y.shape == (2, 1000), f"{name}: bad output shape {y.shape}"
        print(f"OK  {name:35s}  params={params:7.2f}M  out={tuple(y.shape)}")

        if getattr(model, "use_attnres", False):
            with torch.inference_mode():
                weights = model.get_routing_weights(x)
            assert len(weights) == 2 * model.depth + 1, f"{name}: expected one weight map per AttnRes call"
            print(f"    attnres calls={len(weights)}, last alpha shape={tuple(weights[-1].shape)}")

    print("-" * 72)
    check_init_equivalence(device)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
