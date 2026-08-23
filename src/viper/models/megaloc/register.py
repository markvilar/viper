"""
MegaLoc variant catalog — the declarative list of registered variants and the
shared loader that builds each from its checkpoint URL.

This module owns *which* MegaLoc variants exist and *where* their checkpoints
live, separate from the layer and factory code in `model.py`/`backbone.py`/
`aggregation.py`. It is imported for its registration side effect from the
package `__init__`.
"""

import torch

from viper.registry import EmbedderRegistrationEntry, register_embedder_entries
from viper.types import ImageEmbedder

from .model import build_megaloc_from_state_dict

_FAMILY = "megaloc"
_RELEASES = "https://github.com/markvilar/viper/releases/download"


def _load_megaloc_from_url(url: str) -> ImageEmbedder:
    """Loads a MegaLoc variant from a state-dict checkpoint URL. Requires CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MegaLoc but is not available.")
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", weights_only=True
    )
    return build_megaloc_from_state_dict(state_dict).eval().cuda()


_ENTRIES = [
    EmbedderRegistrationEntry(
        key="megaloc",
        family=_FAMILY,
        checkpoint_url=f"{_RELEASES}/megaloc-8448d-v1/megaloc-8448d-v1.pth",
        factory=_load_megaloc_from_url,
    ),
    EmbedderRegistrationEntry(
        key="megaloc-256d-svd-truncated",
        family=_FAMILY,
        checkpoint_url=(
            f"{_RELEASES}/megaloc-svd-truncated-v1.0/megaloc-256d-svd-truncated-v1.pth"
        ),
        factory=_load_megaloc_from_url,
    ),
    EmbedderRegistrationEntry(
        key="megaloc-512d-svd-truncated",
        family=_FAMILY,
        checkpoint_url=(
            f"{_RELEASES}/megaloc-svd-truncated-v1.0/megaloc-512d-svd-truncated-v1.pth"
        ),
        factory=_load_megaloc_from_url,
    ),
    EmbedderRegistrationEntry(
        key="megaloc-1024d-svd-truncated",
        family=_FAMILY,
        checkpoint_url=(
            f"{_RELEASES}/megaloc-svd-truncated-v1.0/megaloc-1024d-svd-truncated-v1.pth"
        ),
        factory=_load_megaloc_from_url,
    ),
]

register_embedder_entries(_ENTRIES)
