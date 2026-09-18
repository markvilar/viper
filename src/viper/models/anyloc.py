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

    def __init__(self, impl: torch.nn.Module, key: str, label: str) -> None:
        """Initializer method."""
        super().__init__()
        self._impl = impl
        self._key = key
        self._label = label
        # NOTE: Add dummy parameter to infer device
        self._param = torch.nn.Parameter(torch.tensor(1.0))

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
        # NOTE: `_impl` is a torch.hub-loaded third-party model whose `vlad`
        # attribute is dynamic and not statically typed.
        impl: typing.Any = self._impl
        return impl.vlad.num_clusters * impl.vlad.desc_dim

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        impl: typing.Any = self._impl
        return {
            "num_clusters": impl.vlad.num_clusters,
            "descriptor_dimensions": impl.vlad.desc_dim,
        }

    @property
    def device(self) -> torch.device:
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
            images = convert_grayscale_batch_to_rgb(images)

        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"

        desired_image_size: tuple[int, int] = calculate_image_size_dinov2(images)
        images_resized: torch.Tensor = resize_image_batch(images, desired_image_size)

        return self._impl(images_resized)


@register_embedder_factory(key="anyloc", label="AnyLoc", family="anyloc")
def load_anyloc(key: str, label: str) -> ImageEmbedder:
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
    wrapper: AnyLocWrapper = AnyLocWrapper(impl=impl, key=key, label=label).eval()
    return wrapper
