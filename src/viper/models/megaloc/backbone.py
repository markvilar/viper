"""
MegaLoc backbone module: DINOv2 ViT-B/14 trunk.
"""

from typing import NamedTuple

import torch.nn as nn
from torch import Tensor

from viper.models.helpers import resize_to_patch_multiple
from viper.models.shared.dinov2 import DINOv2


class DinoFeatures(NamedTuple):
    patches: Tensor    # (B, C, H/14, W/14)
    cls_token: Tensor  # (B, C)


class MegaLocBackboneModule(nn.Module):
    """DINOv2 ViT-B/14 trunk — resizes input to a patch-multiple and returns patch grid + CLS."""

    def __init__(self, dinov2: DINOv2) -> None:
        super().__init__()
        self.dinov2 = dinov2

    @property
    def num_channels(self) -> int:
        return self.dinov2.num_channels

    def forward(self, images: Tensor) -> DinoFeatures:
        images = resize_to_patch_multiple(images, self.dinov2.patch_size)
        patches, cls_token = self.dinov2(images)
        return DinoFeatures(patches, cls_token)
