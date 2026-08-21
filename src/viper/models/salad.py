"""
This module contains a wrapper for the SALAD VPR model. The wrapper wraps SALAD to fit
the ImageEmbedder interface.
The original model repository can be found at: https://github.com/serizba/salad
"""

import typing
import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import calculate_image_size_dinov2
from .helpers import convert_grayscale_batch_to_rgb
from .helpers import resize_image_batch


class SALADWrapper(torch.nn.Module):
    """Class representing a SALAD model."""

    def __init__(self, impl: torch.nn.Module) -> None:
        """Initializer method."""
        super().__init__()
        self.impl = impl

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "salad"

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        global_token_size: int = self.impl.aggregator.token_features[-1].out_features
        local_token_size: int = self.impl.aggregator.cluster_features[-1].out_channels
        local_token_count: int = self.impl.aggregator.score[-1].out_channels
        return local_token_size * local_token_count + global_token_size

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "backbone": "dinov2",
            "backbone_channels": self.impl.backbone.num_channels,
            "aggregator": "salad",
        }

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

        # Resize image heights and widths to multiples of 14
        desired_image_size: tuple[int, int] = calculate_image_size_dinov2(images)
        images_resized: torch.Tensor = resize_image_batch(images, desired_image_size)

        return self.impl.forward(images_resized)


@register_embedder_factory(key="salad", family="salad")
def load_salad() -> ImageEmbedder:
    """Loads a SALAD image embedder model."""
    impl: torch.nn.Module = torch.hub.load("serizba/salad", "dinov2_salad")
    return SALADWrapper(impl=impl)
