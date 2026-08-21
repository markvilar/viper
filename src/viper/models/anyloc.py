"""Module for the Anyloc VPR model."""

import typing
import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb
from .helpers import calculate_image_size_dinov2
from .helpers import resize_image_batch


class AnyLocWrapper(torch.nn.Module):
    """
    Class representing a wrapper for the AnyLoc model.
    """

    def __init__(self, impl: torch.nn.Module) -> None:
        """Initializer method."""
        super().__init__()
        self._impl = impl
        # NOTE: Add dummy parameter to infer device
        self._param = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "anyloc"

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        return self._impl.vlad.num_clusters * self._impl.vlad.desc_dim

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "num_clusters": self._impl.vlad.num_clusters,
            "descriptor_dimensions": self._impl.vlad.desc_dim,
        }

    @property
    def device(self) -> str:
        """Returns the device of the embedder."""
        return next(self.parameters()).device

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embeddes a batch of images.
        :arg images: tensor of shape BxCxHxW
        """
        return self.forward(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forwards a batch of images through the module.

        Arguments:
            images: batch of images - shape BxCxHxW
        Returns:
            batch of image embeddings - shape BxE
        """
        # If the image batch is grayscale, convert to 3 channels
        if images.shape[1] == 1:
            images: torch.Tensor = convert_grayscale_batch_to_rgb(images)

        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"

        desired_image_size: tuple[int, int] = calculate_image_size_dinov2(images)
        images_resized: torch.Tensor = resize_image_batch(images, desired_image_size)

        return self._impl(images_resized)


@register_embedder_factory(key="anyloc", family="anyloc")
def load_anyloc() -> ImageEmbedder:
    """Loads an AnyLoc model."""
    # NOTE: AnyLoc requires CUDA to run, hence we assert
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AnyLoc but is not available.")

    impl: torch.nn.Module = torch.hub.load(
        "AnyLoc/DINO",
        "get_vlad_model",
        backbone="DINOv2",
        domain="unstructured",
        device="cuda",
    )
    wrapper: AnyLocWrapper = AnyLocWrapper(impl=impl).eval()
    return wrapper
