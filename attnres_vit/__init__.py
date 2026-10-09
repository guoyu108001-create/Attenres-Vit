"""AttnRes Vision Transformer — Block AttnRes for ViT/VFM."""

from .vit import VisionTransformerAttnRes
from .factory import register_attnres_vit_models

register_attnres_vit_models()

__all__ = ["VisionTransformerAttnRes", "register_attnres_vit_models"]
