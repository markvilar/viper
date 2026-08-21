"""
Shared neural network modules reused across multiple VPR model implementations.
"""

from .dinov2 import (
    Attention,
    DINOv2,
    LayerScale,
    Mlp,
    PatchEmbedding,
    TransformerBlock,
    build_dinov2_vitb14,
    build_dinov2_vitb14_from_state_dict,
)
from .salad import (
    SALAD,
    L2Norm,
    build_salad,
    build_salad_from_state_dict,
    get_matching_probs,
    log_otp_solver,
)

__all__ = [
    "Attention",
    "DINOv2",
    "LayerScale",
    "Mlp",
    "PatchEmbedding",
    "TransformerBlock",
    "build_dinov2_vitb14",
    "build_dinov2_vitb14_from_state_dict",
    "SALAD",
    "L2Norm",
    "build_salad",
    "build_salad_from_state_dict",
    "get_matching_probs",
    "log_otp_solver",
]
