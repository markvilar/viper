"""
MegaLoc forge: model-specific builders that reshape a MegaLoc model using the
generic decomposition primitives.

A forge takes a fully built model and returns a new model of the same type,
non-destructively rewired around a decomposed component. The generic primitives
in ``viper.model_decomposition`` know nothing about MegaLoc; the knowledge of
*which* component to decompose (the final linear projection in the aggregator)
lives here.
"""

from viper.model_decomposition import truncate_linear
from viper.models.megaloc import MegaLocAggregationModule, MegaLocModel


def forge_megaloc_svd_truncated(model: MegaLocModel, k: int) -> MegaLocModel:
    """
    Build a new MegaLoc model whose descriptor dimension is reduced to ``k`` via
    truncated SVD of the aggregator's final linear projection.

    Only the projection is rebuilt: the backbone and SALAD aggregator are reused
    by reference, so the source model is left unmutated. The truncated projection
    inherits the source layer's device and dtype and keeps a bias iff the source
    does (see ``truncate_linear``). Validation of ``k`` (0 < k <= descriptor_dim)
    is delegated to the primitive.

    Arguments:
        model - source MegaLoc model
        k     - target descriptor dimension, 0 < k <= model.vector_size
    Returns:
        MegaLocModel with vector_size == k, sharing backbone and SALAD with the
        source model
    """
    truncated_linear = truncate_linear(model.aggregator.linear, k)
    aggregator = MegaLocAggregationModule(
        salad=model.aggregator.salad,
        linear=truncated_linear,
    )
    return MegaLocModel(backbone=model.backbone, aggregator=aggregator)
