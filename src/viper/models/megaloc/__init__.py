"""
MegaLoc VPR model — self-contained implementation backed by a bundled checkpoint.
"""

import torch

from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder

from .aggregation import (
    MegaLocAggregationModule,
    build_megaloc_aggregation_from_state_dict,
)
from .backbone import MegaLocBackboneModule, build_megaloc_backbone_from_state_dict
from .model import MegaLocModel, build_megaloc_from_state_dict

__all__ = [
    "MegaLocAggregationModule",
    "MegaLocBackboneModule",
    "MegaLocModel",
    "build_megaloc_aggregation_from_state_dict",
    "build_megaloc_backbone_from_state_dict",
    "build_megaloc_from_state_dict",
]

_FAMILY = "megaloc"

_CHECKPOINT_URL = (
    "https://github.com/markvilar/viper/releases/download"
    "/megaloc-8448d-v1/megaloc-8448d-v1.pth"
)

# SVD-truncated variants published as release `megaloc-svd-truncated-v1.0`,
# keyed by their registry key. See scripts/publish_megaloc_svd_checkpoints.sh.
_SVD_VARIANT_URLS: dict[str, str] = {
    f"{_FAMILY}-{dim}d": (
        "https://github.com/markvilar/viper/releases/download"
        f"/megaloc-svd-truncated-v1.0/megaloc-{dim}d-svd-truncated-v1.0.pth"
    )
    for dim in (256, 512, 1024)
}


def _load_megaloc_from_url(url: str) -> ImageEmbedder:
    """Loads a MegaLoc model from a checkpoint URL. Requires CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MegaLoc but is not available.")
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", weights_only=True
    )
    model = build_megaloc_from_state_dict(state_dict)
    return model.eval().cuda()


@register_embedder_factory(key=_FAMILY, family=_FAMILY)
def load_megaloc() -> ImageEmbedder:
    """Loads the full MegaLoc model from a bundled checkpoint."""
    return _load_megaloc_from_url(_CHECKPOINT_URL)


# Register one factory per SVD-truncated variant, all under the `megaloc` family.
for _key, _url in _SVD_VARIANT_URLS.items():
    register_embedder_factory(key=_key, family=_FAMILY)(
        (lambda url: lambda: _load_megaloc_from_url(url))(_url)
    )
