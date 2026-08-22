"""
MegaLoc VPR model — self-contained implementation backed by a bundled checkpoint.
"""

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

# Imported for its registration side effect: declares the MegaLoc variant catalog.
from . import register as register  # noqa: E402
