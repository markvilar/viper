"""
Model from "MixVPR: Feature Mixing for Visual Place Recognition" - https://arxiv.org/abs/2303.02190
Parts of this code are from https://github.com/amaralibey/MixVPR
"""

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder
from .helpers import convert_grayscale_batch_to_rgb


MIXVPR_URL: str = "https://github.com/markvilar/viper/releases/download/v0.0.1/resnet50_MixVPR_512_channels.256._rows.2.ckpt"
DESCRIPTOR_SIZE: int = 512


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = (
            mlp_ratio  # ratio of the mid projection layer in the mixer block
        )

        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
                for _ in range(self.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()
        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.layer4 = nn.Identity()
        out_channels = 2048
        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

    def forward(self, x1):
        x = self.model.conv1(x1)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


class MixVPRImpl(torch.nn.Module):
    """Class representing a MixVPR implementation."""

    def __init__(self, agg_config: dict[str, typing.Any]):
        """Initializer dunder method."""
        super().__init__()
        self.backbone = ResNet()
        self.aggregator = MixVPR(**agg_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards a batch of images through the model."""
        x: torch.Tensor = transforms.Resize([320, 320], antialias=True)(x)
        x: torch.Tensor = self.backbone(x)
        x: torch.Tensor = self.aggregator(x)
        return x


class MixVPRWrapper(torch.nn.Module):
    """Class representing a MixVPR wrapper."""

    def __init__(self, impl: torch.nn.Module):
        """Initializer dunder method."""
        super().__init__()
        self.impl = impl

    @property
    def name(self) -> str:
        """Returns the name of the embedder."""
        return "mixvpr"

    @property
    def vector_size(self) -> int:
        """Returns the size, i.e. dimensions, of the image embeddings."""
        return DESCRIPTOR_SIZE

    @property
    def embedder_parameters(self) -> dict[str, typing.Any]:
        """Returns the parameters of the embedder."""
        return {
            "backbone": "resnet",
            "aggregator": "mixvpr",
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
        Embeds a batch of images.

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

        return self.impl.forward(images)


@register_embedder_factory(key="mixvpr", family="mixvpr")
def load_mixvpr() -> ImageEmbedder:
    """Loads a MixVPR model."""
    model_config: dict[str, int] = {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": 256,
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": 2,
    }

    state_dict: dict = torch.hub.load_state_dict_from_url(
        MIXVPR_URL,
        map_location="cpu",  # or "cuda"
        progress=True,  # show a download progress bar
        check_hash=False,  # set True if the filename includes a hash
    )

    impl: torch.nn.Module = MixVPRImpl(agg_config=model_config)
    impl.load_state_dict(state_dict)

    return MixVPRWrapper(impl=impl)
