"""
Module for the CliqueMining VPR model.

The original CliqueMining repository can be found at:
https://github.com/serizba/cliquemining/tree/main

The original CliqueMining checkpoint can be found at:
https://drive.google.com/file/d/1B06ysb-Wjb4KDcNrl-7pyj1mJve1jqdk/view
"""

import typing
import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import calculate_image_size_dinov2
from .helpers import convert_grayscale_batch_to_rgb
from .helpers import resize_image_batch


CLIQUE_MINING_CHECKPOINT_URL: str = "https://github.com/markvilar/vpr-model-zoo/releases/download/v0.0.1/cliquemining.ckpt"


class CliqueMiningWrapper(torch.nn.Module):
    """Class representing a wrapper for the CliqueMining model."""

    def __init__(self, impl: torch.nn.Module) -> None:
        """Initializer method."""
        super().__init__()
        self.impl = impl

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "cliquemining"

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
        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"

        # If the image batch is grayscale, convert to 3 channels
        if images.shape[1] == 1:
            images: torch.Tensor = convert_grayscale_batch_to_rgb(images)

        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"

        # Resize image heights and widths to multiples of 14
        desired_image_size: tuple[int, int] = calculate_image_size_dinov2(images)
        images_resized: torch.Tensor = resize_image_batch(images, desired_image_size)

        return self.impl.forward(images_resized)


@register_embedder_factory(key="cliquemining")
def load_clique_mining() -> ImageEmbedder:
    """
    Loads a CliqueMining wrapper by downloading SALAD from torch hub.
    """
    # NOTE: CliqueMining requires CUDA to run, hence we assert
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CliqueMining but is not available.")

    # Load the SALAD model (which has the same architecture as CliqueMining)
    # and then load clique-mining's weights
    impl: torch.nn.Module = torch.hub.load("serizba/salad", "dinov2_salad")

    # Download the model checkpoint
    checkpoint: dict = torch.hub.load_state_dict_from_url(
        CLIQUE_MINING_CHECKPOINT_URL,
        map_location="cpu",  # or "cuda"
        progress=True,  # show a download progress bar
        check_hash=False,  # set True if the filename includes a hash
    )

    assert "state_dict" in checkpoint, "missing checkpoint key: 'state_dict'"

    state_dict: dict = checkpoint.get("state_dict")
    impl.load_state_dict(state_dict)

    return CliqueMiningWrapper(impl=impl)
