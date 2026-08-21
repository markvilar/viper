"""
MegaLoc VPR model — self-contained implementation backed by a bundled checkpoint.
"""

import torch
import torch.nn as nn

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder

from viper.models.shared.dinov2 import build_dinov2_vitb14
from viper.models.shared.salad import build_salad

from .aggregation import MegaLocAggregationModule
from .backbone import MegaLocBackboneModule
from .model import MegaLocModel

__all__ = [
    "MegaLocAggregationModule",
    "MegaLocBackboneModule",
    "MegaLocModel",
]

_CHECKPOINT_URL = (
    "https://github.com/markvilar/viper/releases/download"
    "/megaloc-8448d-v1/megaloc-8448d-v1.pth"
)

# SALAD out_dim = 64 * 256 + 256 = 16640; linear projects to 8448-D descriptor
_SALAD_OUT_DIM = 16640
_DESCRIPTOR_DIM = 8448


@register_embedder_factory(key="megaloc")
def load_megaloc() -> ImageEmbedder:
    """Loads MegaLoc from a bundled checkpoint hosted in this repository."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MegaLoc but is not available.")
    state = torch.hub.load_state_dict_from_url(
        _CHECKPOINT_URL, map_location="cpu", weights_only=True
    )
    model = MegaLocModel(
        backbone=MegaLocBackboneModule(build_dinov2_vitb14()),
        aggregator=MegaLocAggregationModule(
            salad=build_salad(dropout=0.0),
            linear=nn.Linear(_SALAD_OUT_DIM, _DESCRIPTOR_DIM),
        ),
    )
    model.load_state_dict(state, strict=True)
    return model.eval().cuda()
