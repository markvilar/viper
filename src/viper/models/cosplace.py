"""
Module for the CosPlace image embedder.
"""

import typing
import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb


class CosPlaceWrapper(torch.nn.Module):
    """
    Class representing a wrapper for the Cosplace VPR model.
    The wrapper implements the image embedder interface for the Cosplace model.
    """

    def __init__(
        self, impl: torch.nn.Module, backbone: str, descriptor_size: int
    ) -> None:
        """Initializer method."""
        super().__init__()
        self.impl = impl
        self.backbone = backbone
        self.descriptor_size = descriptor_size

        linear_aggregation_layers: list[torch.nn.Linear] = [
            layer
            for layer in self.impl.aggregation
            if isinstance(layer, torch.nn.Linear)
        ]
        assert linear_aggregation_layers[-1].out_features == self.descriptor_size, (
            "output features of aggregation layer does not match descriptor size"
        )

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "cosplace"

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        return self.descriptor_size

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {"backbone": self.backbone, "descriptor_size": self.descriptor_size}

    @property
    def device(self) -> str:
        """Returns the device of the embedder."""
        return next(self.parameters()).device

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of images.

        Arguments:
            images: batch of images, shape BxCxHxW, dtype float
        Returns:
            batch of image embeddings, shape BxE
        """
        return self.forward(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of images. Handles both RGB and grayscale images, both of shape
        BxCxHxW.

        Arguments:
            images: batch of images, shape BxCxHxW, dtype float
        Returns:
            batch of image embeddings, shape BxE
        """
        assert images.dim() == 4, (
            f"invalid batch dimensions: expected 4, got {images.dim()}"
        )
        # If the image batch is grayscale, convert to 3 channels
        if images.shape[1] == 1:
            images: torch.Tensor = convert_grayscale_batch_to_rgb(images)
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"
        return self.impl.forward(images)


@register_embedder_factory(key="cosplace", family="cosplace")
def load_cosplace() -> ImageEmbedder:
    """
    Loads a CosPlace model from torch hub and adds it to a wrapper.
    """
    BACKBONE: str = "ResNet50"
    DESCRIPTORS_DIMENSION: int = 2048
    impl: torch.nn.Module = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone=BACKBONE,
        fc_output_dim=DESCRIPTORS_DIMENSION,
    )
    return CosPlaceWrapper(
        impl=impl,
        backbone=BACKBONE,
        descriptor_size=DESCRIPTORS_DIMENSION,
    )
