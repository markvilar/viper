"""
MegaLoc aggregation module: SALAD -> Linear -> L2Norm.
"""

import torch.nn as nn
from torch import Tensor

from viper.models.shared.salad import L2Norm, SALAD

from .backbone import DinoFeatures


class MegaLocAggregationModule(nn.Module):
    """SALAD aggregator followed by a linear projection and L2 normalisation."""

    def __init__(self, salad: SALAD, linear: nn.Linear) -> None:
        super().__init__()
        self.salad = salad
        self.linear = linear
        self.norm = L2Norm()

    @property
    def descriptor_dim(self) -> int:
        return self.linear.out_features

    def forward(self, features: DinoFeatures) -> Tensor:
        return self.norm(self.linear(self.salad(features)))
