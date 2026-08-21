"""
Generic, model-agnostic primitives for decomposing and remapping the building
blocks of pretrained embedders.

Operates on raw tensors and ``nn.Module``s with no knowledge of any specific
model; the model-specific builders that apply these primitives to a given
architecture live with their respective models.
"""

from .linear import truncate_linear, truncate_linear_svd

__all__ = [
    "truncate_linear",
    "truncate_linear_svd",
]
