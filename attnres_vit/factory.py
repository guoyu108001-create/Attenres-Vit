"""Register AttnRes ViT models with timm."""

from __future__ import annotations

from timm.models import register_model

from .vit import VisionTransformerAttnRes


def _build(
    depth: int,
    embed_dim: int,
    num_heads: int,
    use_attnres: bool,
    num_blocks: int = 4,
    drop_path_rate: float = 0.1,
    pretrained: bool = False,
    **kwargs,
):
    if pretrained:
        raise NotImplementedError("No pretrained AttnRes ViT weights yet.")

    # timm passes registry metadata kwargs; ignore unknown keys.
    allowed = {
        "img_size", "patch_size", "in_chans", "num_classes", "mlp_ratio",
        "qkv_bias", "proj_bias", "drop_rate", "attn_drop_rate", "init_values", "global_pool",
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    model = VisionTransformerAttnRes(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_attnres=use_attnres,
        num_blocks=num_blocks,
        drop_path_rate=drop_path_rate,
        **model_kwargs,
    )
    return model


def register_attnres_vit_models() -> None:
    """Register all Phase-0 model variants."""

    @register_model
    def attnres_vit_d12_base(pretrained=False, **kwargs):
        """ViT-B/16 depth=12 baseline (standard residuals)."""
        return _build(12, 768, 12, use_attnres=False, drop_path_rate=0.1, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d12_base_attnres(pretrained=False, **kwargs):
        """ViT-B/16 depth=12 + Block AttnRes (4 blocks)."""
        return _build(12, 768, 12, use_attnres=True, num_blocks=4, drop_path_rate=0.1, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d24_base(pretrained=False, **kwargs):
        """ViT depth=24, dim=768 baseline."""
        return _build(24, 768, 12, use_attnres=False, drop_path_rate=0.2, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d24_base_attnres(pretrained=False, **kwargs):
        """ViT depth=24, dim=768 + Block AttnRes (8 blocks)."""
        return _build(24, 768, 12, use_attnres=True, num_blocks=8, drop_path_rate=0.2, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d48_narrow(pretrained=False, **kwargs):
        """Deep-narrow ViT: depth=48, dim=512 baseline."""
        return _build(48, 512, 8, use_attnres=False, drop_path_rate=0.3, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d48_narrow_attnres(pretrained=False, **kwargs):
        """Deep-narrow ViT: depth=48, dim=512 + Block AttnRes (8 blocks)."""
        return _build(48, 512, 8, use_attnres=True, num_blocks=8, drop_path_rate=0.3, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d48_wide(pretrained=False, **kwargs):
        """Deep-wide ViT: depth=48, dim=768 baseline."""
        return _build(48, 768, 12, use_attnres=False, drop_path_rate=0.3, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_d48_wide_attnres(pretrained=False, **kwargs):
        """Deep-wide ViT: depth=48, dim=768 + Block AttnRes (8 blocks)."""
        return _build(48, 768, 12, use_attnres=True, num_blocks=8, drop_path_rate=0.3, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_tiny(pretrained=False, **kwargs):
        """Smoke-test model: depth=6, dim=192."""
        return _build(6, 192, 3, use_attnres=False, num_blocks=3, drop_path_rate=0.05, pretrained=pretrained, **kwargs)

    @register_model
    def attnres_vit_tiny_attnres(pretrained=False, **kwargs):
        """Smoke-test model: depth=6, dim=192 + Block AttnRes."""
        return _build(6, 192, 3, use_attnres=True, num_blocks=3, drop_path_rate=0.05, pretrained=pretrained, **kwargs)
