"""
MegaLoc composite model.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from viper.models.helpers import (
    convert_grayscale_batch_to_rgb,
    extract_submodule_state_dict,
)

from .aggregation import (
    MegaLocAggregationModule,
    build_megaloc_aggregation_from_state_dict,
)
from .backbone import MegaLocBackboneModule, build_megaloc_backbone_from_state_dict


class MegaLocModel(nn.Module):
    """Composite MegaLoc embedder. Satisfies the ImageEmbedder protocol."""

    def __init__(
        self,
        backbone: MegaLocBackboneModule,
        aggregator: MegaLocAggregationModule,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    @property
    def name(self) -> str:
        return "megaloc"

    @property
    def vector_size(self) -> int:
        return self.aggregator.descriptor_dim

    @property
    def embedder_parameters(self) -> dict[str, Any]:
        return {
            "backbone": "dinov2",
            "backbone_channels": self.backbone.num_channels,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, images: Tensor) -> Tensor:
        if images.dim() != 4:
            raise ValueError(
                f"expected a 4D batch (B, C, H, W), got shape {tuple(images.shape)}"
            )
        if images.shape[1] == 1:
            images = convert_grayscale_batch_to_rgb(images)
        if images.shape[1] != 3:
            raise ValueError(
                f"expected 1 or 3 channels, got {images.shape[1]}"
            )
        return self.aggregator(self.backbone(images))


def build_megaloc_from_state_dict(state_dict: Mapping[str, Tensor]) -> MegaLocModel:
    """
    Build a MegaLoc model from a full checkpoint state dict.

    Components are built and loaded bottom-up from their respective slices of the
    state dict and then composed — no dimension is hardcoded and no load-time
    remapping is performed (the checkpoint keys are pre-aligned to the module
    hierarchy: "backbone.*" and "aggregator.*").

    Arguments:
        state_dict - full MegaLoc state dict
    Returns:
        MegaLocModel with all weights loaded
    """
    backbone = build_megaloc_backbone_from_state_dict(
        extract_submodule_state_dict(state_dict, "backbone.")
    )
    aggregator = build_megaloc_aggregation_from_state_dict(
        extract_submodule_state_dict(state_dict, "aggregator.")
    )
    return MegaLocModel(backbone=backbone, aggregator=aggregator)
