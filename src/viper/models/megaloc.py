"""
Module for the Megaloc VPR model.

The original MegaLoc repository can be found at the following URL:
https://github.com/gmberton/MegaLoc/tree/main
"""

import typing
import torch
import torch.nn as nn

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb
from .helpers import calculate_image_size_dinov2
from .helpers import resize_image_batch


class MegaLocWrapper(nn.Module):
    """Class representing a wrapper for the MegaLoc model."""

    def __init__(self, impl: torch.nn.Module) -> None:
        """Initializer method."""
        super().__init__()
        self.impl = impl

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "megaloc"

    @property
    def vector_size(self) -> int:
        """Returns the embedding size, i.e. the dimensions of the image embedding."""
        return self.impl.feat_dim

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "backbone": "dinov2",
            "backbone_channels": self.impl.backbone.num_channels,
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
        # If the image batch is grayscale, convert to 3 channels
        if images.shape[1] == 1:
            images: torch.Tensor = convert_grayscale_batch_to_rgb(images)

        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"

        # Resize image heights and widths to multiples of 14
        desired_image_size: tuple[int, int] = calculate_image_size_dinov2(images)
        images_resized: torch.Tensor = resize_image_batch(images, desired_image_size)

        return self.impl.forward(images_resized)


@register_embedder_factory(key="megaloc")
def load_megaloc() -> ImageEmbedder:
    """Loads a MegaLoc model from torch hub."""
    # NOTE: MegaLoc requires CUDA to run, hence we assert
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MegaLoc but is not available.")

    impl: torch.nn.Module = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    wrapper: MegaLocWrapper = MegaLocWrapper(impl=impl).eval()
    return wrapper
