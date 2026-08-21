"""
Model-specific builders ("forges") that reshape pretrained embedders using the
generic primitives in ``viper.model_decomposition``.

Where the primitives operate on raw tensors and modules with no knowledge of any
architecture, a forge encodes which component of a specific model to decompose
and returns a new model of the same type, non-destructively rewired around it.
"""

from .megaloc import forge_megaloc_svd_truncated

__all__ = [
    "forge_megaloc_svd_truncated",
]
