"""Module for the EigenPlaces VPR model."""

import typing
import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb

BACKBONE: str = "ResNet50"
DESCRIPTORS_DIMENSION: int = 2048


class EigenPlacesWrapper(torch.nn.Module):
    """Class representing an EigenPlaces wrapper."""

    def __init__(self, impl: torch.nn.Module, key: str, label: str) -> None:
        """Initializer method."""
        super().__init__()
        self.impl = impl
        self._key = key
        self._label = label

    @property
    def key(self) -> str:
        """Returns the registry lookup key of the embedder."""
        return self._key

    @property
    def label(self) -> str:
        """Returns the presentation label of the embedder."""
        return self._label

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        return DESCRIPTORS_DIMENSION

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "backbone": BACKBONE,
        }

    @property
    def device(self) -> torch.device:
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
        Embeds a batch of images.

        Arguments:
            images: batch of images, shape BxCxHxW, dtype float
        Returns:
            batch of image embeddings, shape BxE
        """
        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
        if images.shape[1] == 1:
            images = convert_grayscale_batch_to_rgb(images)
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"
        return self.impl.forward(images)


@register_embedder_factory(key="eigenplaces", label="EigenPlaces", family="eigenplaces")
def load_eigenplaces(key: str, label: str) -> ImageEmbedder:
    """Loads an EigenPlaces model."""
    impl: torch.nn.Module = torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone=BACKBONE,
        fc_output_dim=DESCRIPTORS_DIMENSION,
    )
    return EigenPlacesWrapper(impl=impl, key=key, label=label)
