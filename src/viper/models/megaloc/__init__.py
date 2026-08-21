"""
MegaLoc VPR model — self-contained implementation backed by a bundled checkpoint.
"""

import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder

from .aggregation import (
    MegaLocAggregationModule,
    build_megaloc_aggregation_from_state_dict,
)
from .backbone import MegaLocBackboneModule, build_megaloc_backbone_from_state_dict
from .model import MegaLocModel, build_megaloc_from_state_dict

__all__ = [
    "MegaLocAggregationModule",
    "MegaLocBackboneModule",
    "MegaLocModel",
    "build_megaloc_aggregation_from_state_dict",
    "build_megaloc_backbone_from_state_dict",
    "build_megaloc_from_state_dict",
]

_CHECKPOINT_URL = (
    "https://github.com/markvilar/viper/releases/download"
    "/megaloc-8448d-v1/megaloc-8448d-v1.pth"
)


@register_embedder_factory(key="megaloc")
def load_megaloc() -> ImageEmbedder:
    """Loads MegaLoc from a bundled checkpoint hosted in this repository."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MegaLoc but is not available.")
    state_dict = torch.hub.load_state_dict_from_url(
        _CHECKPOINT_URL, map_location="cpu", weights_only=True
    )
    model = build_megaloc_from_state_dict(state_dict)
    return model.eval().cuda()
