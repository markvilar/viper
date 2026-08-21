"""
MegaLoc composite model.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from viper.models.helpers import convert_grayscale_batch_to_rgb

from .aggregation import MegaLocAggregationModule
from .backbone import MegaLocBackboneModule


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
        if images.shape[1] == 1:
            images = convert_grayscale_batch_to_rgb(images)
        assert images.dim() == 4, f"expected (B, C, H, W), got {images.shape}"
        assert images.shape[1] == 3, f"expected 3 channels, got {images.shape[1]}"
        return self.aggregator(self.backbone(images))
