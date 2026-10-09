"""Attention Residuals modules for Vision Transformers.

Reference: Kimi Team, "Attention Residuals" (arXiv:2603.15031)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """Root-mean-square normalization (AttnRes paper default)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _inv_rms(v: torch.Tensor, eps: float) -> torch.Tensor:
    return v.pow(2).mean(dim=-1).add(eps).rsqrt()


def inv_rms(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """1 / RMS(v) over the last dim, [B, T] (autocast runs pow in fp32, as in RMSNorm). Under
    autograd it is recomputed in backward, so no fp32 copy of a half-precision source stays alive."""
    if torch.is_grad_enabled():
        return checkpoint(_inv_rms, v, eps, use_reentrant=False)
    return _inv_rms(v, eps)


class BlockAttnRes(nn.Module):
    """Softmax attention over depth sources with a learned pseudo-query (paper Fig. 2).

    Sources are [b_0 (token embedding), b_1, ..., b_{n-1}] plus, after the first sublayer
    of a block, the partial sum b_n^i; each b is a sum of sublayer outputs. Keys are
    RMSNorm(source); the pseudo-query is zero-initialized (paper Sec. 5), so the weights
    start uniform.

    Same function as stacking the sources and applying ``self.norm``, computed without the
    [S, B, T, D] stack: the RMSNorm gain is folded into the query, w . (g * v * r) =
    ((w * g) . v) * r, and r = 1 / RMS(v) is computed once per source by the caller.
    """

    def __init__(self, hidden_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        sources: List[torch.Tensor],
        inv_rms: List[torch.Tensor],
        return_alpha: bool = False,
    ):
        """Aggregate sources (each [B, T, D]) given their ``inv_rms`` ([B, T]); optionally
        also return the weights [S, B, T]."""
        q = self.proj.weight.squeeze(0) * self.norm.weight
        logits = torch.stack([(v * q).sum(dim=-1) * r for v, r in zip(sources, inv_rms)])
        alpha = logits.softmax(dim=0)
        h = alpha[0].unsqueeze(-1) * sources[0]
        for a, v in zip(alpha[1:], sources[1:]):
            h = h + a.unsqueeze(-1) * v
        return (h, alpha) if return_alpha else h
