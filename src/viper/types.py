"""Module for image embedder interface."""

import typing
import torch

from collections.abc import Callable


class ImageEmbedder(typing.Protocol):
    """Class representing the interface for an image embedder."""

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        ...

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        ...

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        ...

    @property
    def device(self) -> str:
        """Returns the device of the embedder."""
        ...

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of images.

        Arguments:
            images: batch of images, shape BxCxHxW, dtype float
        Returns:
            batch of image embeddings, shape BxE
        """
        ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of images.

        Arguments:
            images: batch of images, shape BxCxHxW, dtype float
        Returns:
            batch of image embeddings, shape BxE
        """
        ...


type ImageEmbedderFactory = Callable[[...], ImageEmbedder]
