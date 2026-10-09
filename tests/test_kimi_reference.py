#!/usr/bin/env python3
"""AttnRes ViT vs a verbatim port of Kimi's Fig. 2 pseudo-code (arXiv:2603.15031).

float64 with random (non-zero) pseudo-queries and RMSNorm gains, so it checks the wiring and the
stack-free computation, not just the uniform-weight case at init. CPU, ~1 min.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "timm"))

import torch
import attnres_vit  # noqa: F401
from timm.models import create_model


def block_attn_res(blocks, partial_block, proj, norm):
    """Kimi Fig. 2, verbatim."""
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(), K)
    return torch.einsum("n b t, n b t d -> b t d", logits.softmax(0), V)


def kimi_reference(model, x):
    """Kimi Fig. 2 forward: per-layer pre-attn / pre-MLP block_attn_res, 0-indexed layer_number."""
    mods = [lambda srcs, m=m: block_attn_res(srcs[:-1], srcs[-1], m.proj, m.norm) for m in model.attn_res]
    h = model.norm_pre(model._pos_embed(model.patch_embed(x)))
    blocks, partial = [], h  # layer 0 appends the embedding as b_0
    for l, blk in enumerate(model.blocks):
        hh = mods[2 * l](blocks + [partial])
        if l % model.block_size == 0:
            blocks.append(partial)
            partial = None
        attn_out = blk.attn_branch(hh)
        partial = attn_out if partial is None else partial + attn_out
        hh = mods[2 * l + 1](blocks + [partial])
        partial = partial + blk.mlp_branch(hh)
    return model.norm(mods[2 * model.depth](blocks + [partial]))


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224, dtype=torch.float64)
    for name in ["attnres_vit_tiny_attnres", "attnres_vit_d12_base_attnres", "attnres_vit_d24_base_attnres"]:
        m = create_model(name).double().eval()
        assert all(float(mod.proj.weight.detach().abs().max()) == 0.0 for mod in m.attn_res), "queries not zero-initialized"
        with torch.no_grad():
            for mod in m.attn_res:  # random queries / gains: non-uniform depth weights
                mod.proj.weight.normal_(std=0.2)
                mod.norm.weight.uniform_(0.5, 1.5)
            ours, ref = m.forward_features(x), kimi_reference(m, x)
        rel = ((ours - ref).norm() / ref.norm()).item()
        assert rel < 1e-10, f"{name}: differs from Kimi Fig. 2 (rel {rel:.2e})"
        print(f"OK  {name:32s} == Kimi Fig. 2 port (float64 rel diff {rel:.1e})")
    print("Kimi reference test passed.")


if __name__ == "__main__":
    main()
