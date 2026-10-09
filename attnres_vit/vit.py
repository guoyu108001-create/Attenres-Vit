"""Vision Transformer with optional Block AttnRes."""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import (
    Attention,
    DropPath,
    LayerScale,
    Mlp,
    PatchDropout,
    PatchEmbed,
    calculate_drop_path_rates,
    trunc_normal_,
)
from timm.models.vision_transformer import global_pool_nlc

from .modules import BlockAttnRes, inv_rms


class ViTBlock(nn.Module):
    """Standard PreNorm ViT block (intra-block additive residuals)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            bias=proj_bias,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path1(self.ls1(self.attn(self.norm1(x))))

    def mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_branch(x)
        x = x + self.mlp_branch(x)
        return x


class VisionTransformerAttnRes(nn.Module):
    """ViT backbone with optional Block AttnRes (Kimi Team, arXiv:2603.15031).

    With AttnRes, every sublayer (attention and MLP) reads a softmax mixture of
    [token embedding, completed block sums, current partial sum] instead of the
    residual stream, and the final norm reads a mixture of all block sums.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = None,
        use_attnres: bool = False,
        num_blocks: int = 4,
        global_pool: str = "token",
    ) -> None:
        super().__init__()
        if use_attnres and depth % num_blocks != 0:
            raise ValueError(f"depth ({depth}) must be divisible by num_blocks ({num_blocks})")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_attnres = use_attnres
        self.num_blocks = num_blocks
        self.block_size = depth // num_blocks
        self.global_pool = global_pool
        self.num_prefix_tokens = 1

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

        dpr = calculate_drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        if use_attnres:
            # One per sublayer (attn, MLP of each layer) plus one for the final output.
            self.attn_res = nn.ModuleList([
                BlockAttnRes(embed_dim) for _ in range(2 * depth + 1)
            ])
        else:
            self.attn_res = None

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        if self.attn_res is not None:
            # Pseudo-queries must start at zero (uniform depth weights); _init_weights resets them.
            for mod in self.attn_res:
                nn.init.zeros_(mod.proj.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def _forward_attnres(
        self, x: torch.Tensor, alphas: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Block AttnRes forward (paper Eq. 5-6, Fig. 2); appends weights to ``alphas`` if given."""

        def mix(idx: int, sources: List[torch.Tensor], rs: List[torch.Tensor]) -> torch.Tensor:
            if alphas is None:
                return self.attn_res[idx](sources, rs)
            h, alpha = self.attn_res[idx](sources, rs, return_alpha=True)
            alphas.append(alpha.detach())
            return h

        eps = self.attn_res[0].norm.eps
        # b_0 = token embedding, then one sum of sublayer outputs per completed block;
        # rs holds each source's 1 / RMS, computed once and shared by all later calls.
        blocks, rs = [x], [inv_rms(x, eps)]
        partial = r_partial = None  # sum of sublayer outputs in the current block
        for i, blk in enumerate(self.blocks):
            if partial is None:
                h = mix(2 * i, blocks, rs)
            else:
                h = mix(2 * i, blocks + [partial], rs + [r_partial])
            out = blk.attn_branch(h)
            # Accumulate in the embedding's dtype (fp32 under AMP), like the baseline residual stream;
            # summing fp16 sublayer outputs in fp16 overflowed in training (E02, 2026-10-09).
            partial = out.to(x.dtype) if partial is None else partial + out
            r_partial = inv_rms(partial, eps)
            h = mix(2 * i + 1, blocks + [partial], rs + [r_partial])
            partial = partial + blk.mlp_branch(h)
            r_partial = inv_rms(partial, eps)
            if (i + 1) % self.block_size == 0:
                blocks.append(partial)
                rs.append(r_partial)
                partial = r_partial = None
        return mix(2 * self.depth, blocks, rs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        if not self.use_attnres:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = self._forward_attnres(x)

        return self.norm(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.global_pool == "token":
            x = x[:, 0]
        else:
            x = global_pool_nlc(x, pool_type=self.global_pool, num_prefix_tokens=self.num_prefix_tokens)
        if pre_logits:
            return x
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)

    def get_routing_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return depth softmax weights [S, B, T] per AttnRes call (pre-attn, pre-MLP of
        each layer, then the final output), for visualization."""
        if not self.use_attnres:
            return []

        weights: List[torch.Tensor] = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        self._forward_attnres(x, alphas=weights)
        return weights
