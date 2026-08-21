"""
MegaLoc aggregation module: SALAD -> Linear -> L2Norm.
"""

from collections.abc import Mapping

import torch.nn as nn
from torch import Tensor

from viper.models.helpers import extract_submodule_state_dict
from viper.models.shared.salad import L2Norm, SALAD, build_salad_from_state_dict

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


def build_megaloc_aggregation_from_state_dict(
    state_dict: Mapping[str, Tensor],
) -> MegaLocAggregationModule:
    """
    Build a MegaLoc aggregation module from a state dict.

    The SALAD aggregator is built via the shared factory; the linear projection's
    input and output dimensions are derived from its weight shape
    (out_features, in_features), so no dimension is hardcoded. Expects keys
    relative to the aggregation module (i.e. the "aggregator." prefix already
    stripped), so SALAD weights are under "salad." and the projection under
    "linear.".

    Arguments:
        state_dict - aggregation state dict
    Returns:
        MegaLocAggregationModule with weights loaded
    """
    salad = build_salad_from_state_dict(
        extract_submodule_state_dict(state_dict, "salad.")
    )
    linear_state_dict = extract_submodule_state_dict(state_dict, "linear.")
    out_features, in_features = linear_state_dict["weight"].shape
    linear = nn.Linear(in_features, out_features)
    linear.load_state_dict(linear_state_dict, strict=True)
    return MegaLocAggregationModule(salad=salad, linear=linear)
