"""
Model from "NetVLAD: CNN architecture for weakly supervised place recognition"
https://arxiv.org/abs/1511.07247
Parts of this code are from https://github.com/cvg/Hierarchical-Localization
"""

import subprocess
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pathlib import Path
from scipy.io import loadmat

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb


EPS = 1e-6


class NetVLADLayer(nn.Module):
    """Class representing a NetVLAD layer."""

    def __init__(
        self,
        input_dim: int = 512,
        num_clusters: int = 64,
        score_bias: bool = False,
        intranorm: bool = True,
    ) -> None:
        """Constructor method."""
        super().__init__()
        self.score_proj: nn.Conv1d = nn.Conv1d(
            input_dim, num_clusters, kernel_size=1, bias=score_bias
        )
        centers: nn.parameter.Parameter = nn.parameter.Parameter(
            torch.empty([input_dim, num_clusters])
        )
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self._intranorm: bool = intranorm
        self._num_cluters: int = num_clusters
        self._output_dim: int = input_dim * num_clusters

    @property
    def intranorm(self) -> bool:
        """The intranorm property."""
        return self._intranorm

    @property
    def cluster_count(self) -> int:
        """Returns the number of clusters in the layer."""
        return self._num_cluters

    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the NetVLAD layer."""
        return self._output_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forwards a batch of images through the tensor.
        :arg batch: image batch of shape BxCxHxW
        """
        batch_size: int = batch.size(0)
        scores: torch.Tensor = self.score_proj(batch)
        scores: torch.Tensor = F.softmax(scores, dim=1)

        diff: torch.Tensor = batch.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(
            -1
        )
        desc: torch.Tensor = (scores.unsqueeze(1) * diff).sum(dim=-1)

        if self.intranorm:
            # From the official MATLAB implementation.
            desc: torch.Tensor = F.normalize(desc, dim=1)
        desc: torch.Tensor = desc.view(batch_size, -1)
        desc: torch.Tensor = F.normalize(desc, dim=1)
        return desc


class NetVLAD(torch.nn.Module):
    """Class representing a NetVLAD torch module."""

    def __init__(self, descriptors_dimension: int) -> None:
        super().__init__()

        assert descriptors_dimension in [4096, 32768]

        # TODO: Create a factory function
        whiten: bool = descriptors_dimension == 4096
        model_name: str = "VGG16-NetVLAD-Pitts30K"
        dir_models: dict[str, str] = {
            "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",
            "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",
        }

        # Download the checkpoint.
        checkpoint: Path = Path(torch.hub.get_dir(), "netvlad", model_name + ".mat")
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True, parents=True)
            link = dir_models[model_name]
            cmd = ["wget", link, "-O", str(checkpoint)]
            subprocess.run(cmd, check=True)

        # Create the network.
        # Remove classification head.
        backbone = list(models.vgg16().children())[0]

        # Remove last ReLU + MaxPool2d.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.netvlad_layer = NetVLADLayer()

        if whiten:
            self.whiten = nn.Linear(self.netvlad_layer.output_dim, 4096)
        else:
            self.whiten = None

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        mat = loadmat(checkpoint, struct_as_record=False, squeeze_me=True)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = mat["net"].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -mat["net"].layers[30].weights[1]  # D x K
        # Prepare for PyTorch - make sure it is float32 and has right shape.
        # score_w should have shape K x D x 1
        # center_w should have shape D x K
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        # Update layer weights.
        self.netvlad_layer.score_proj.weight = nn.Parameter(score_w)
        self.netvlad_layer.centers = nn.Parameter(center_w)

        # Whitening weights.
        if self.whiten:
            w = mat["net"].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
            b = mat["net"].layers[33].weights[1]  # Shape: OUT
            # Prepare for PyTorch - make sure it is float32 and has right shape
            w = torch.tensor(w).float().squeeze().permute([1, 0])  # OUT x IN
            b = torch.tensor(b.squeeze()).float()  # Shape: OUT
            # Update layer weights.
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing parameters.
        self.preprocess = {
            "mean": mat["net"].meta.normalization.averageImage[0, 0],
            "std": np.array([1, 1, 1], dtype=np.float32),
        }

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "netvlad"

    @property
    def vector_size(self) -> int:
        """Returns the embedding size, i.e. the dimensions of the image embedding."""
        if self.whiten:
            return self.whiten.out_features
        else:
            return self.netvlad_layer.output_dim

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "backbone": "vgg16",
            "cluster_count": self.netvlad_layer.cluster_count,
        }

    @property
    def device(self) -> str:
        """Returns the device of the embedder."""
        return next(self.parameters()).device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forwards a batch of images through the model and returns embeddings.
        Assumes that the input images are normalized, i.e. between 0.0 and 1.0.
        """

        assert images.max() <= 1.0, "image values must be below 1.0"
        assert images.min() >= 0.0, "image values must be above 0.0"

        images: torch.Tensor = denormalize_images(images)

        # If the image batch is grayscale, convert to 3 channels
        if images.shape[1] == 1:
            images: torch.Tensor = convert_grayscale_batch_to_rgb(images)

        assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
        assert images.shape[1] == 3, f"invalid image batch channels: {images.shape[1]}"
        assert images.min() >= -EPS and images.max() <= 1 + EPS, (
            "images must be between 0.0 and 1.0"
        )

        # Input should be 0-255.
        images = torch.clamp(images * 255, 0.0, 255.0)

        mean = self.preprocess["mean"]
        std = self.preprocess["std"]

        images: torch.Tensor = images - images.new_tensor(mean).view(1, -1, 1, 1)
        images: torch.Tensor = images / images.new_tensor(std).view(1, -1, 1, 1)

        # Feature extraction.
        descriptors: torch.Tensor = self.backbone(images)
        b, c, _, _ = descriptors.size()
        descriptors: torch.Tensor = descriptors.view(b, c, -1)

        # NetVLAD layer.
        descriptors: torch.Tensor = F.normalize(
            descriptors, dim=1
        )  # Pre-normalization.
        descriptors: torch.Tenosr = self.netvlad_layer(descriptors)

        # Whiten if needed.
        if hasattr(self, "whiten"):
            # Whiten descriptors and perform final L2 normalization.
            descriptors: torch.Tensor = self.whiten(descriptors)
            descriptors: torch.Tensor = F.normalize(descriptors, dim=1)

        return descriptors


IMAGE_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGE_STD: list[float] = [0.229, 0.224, 0.225]


def denormalize_images(
    batch: torch.Tensor,
    mean: list[float] = IMAGE_MEAN,
    std: list[float] = IMAGE_STD,
) -> torch.Tensor:
    """
    Denormalize a batch of images by multiplying with the standard deviation and
    adding a mean.

    :return: tensor of shape BxCxHxW
    """
    # 3, H, W, B
    batch: torch.Tensor = batch.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(batch, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(batch, 0, 1).permute(3, 0, 1, 2)


@register_embedder_factory(key="netvlad")
def load_netvlad() -> ImageEmbedder:
    """
    Creates a wrapper for a NetVLAD model.
    """
    # TODO: Move weight loading etc. from NetVLAD to this function!
    return NetVLAD(descriptors_dimension=4096)
