"""
Tests for the MegaLoc forge, verifying that SVD truncation reduces the descriptor
dimension, leaves the source model unmutated, and reuses the backbone and SALAD
aggregator by reference.

The forge performs no forward pass, so these are structural assertions only: no
checkpoint or CUDA device is required. Lightweight ``nn.Module`` stand-ins take
the place of the real backbone and SALAD aggregator, whose identities are all the
forge depends on.
"""

import torch.nn as nn

from viper.forge import forge_megaloc_svd_truncated
from viper.models.megaloc import MegaLocAggregationModule, MegaLocModel


def _build_megaloc(descriptor_dim: int, feature_dim: int = 16) -> MegaLocModel:
    backbone = nn.Identity()
    salad = nn.Identity()
    linear = nn.Linear(feature_dim, descriptor_dim)
    aggregator = MegaLocAggregationModule(salad=salad, linear=linear)
    return MegaLocModel(backbone=backbone, aggregator=aggregator)


def test_forge_megaloc_svd_truncated_reduces_descriptor_dim() -> None:
    # Arrange
    model = _build_megaloc(descriptor_dim=12)

    # Act
    forged = forge_megaloc_svd_truncated(model, k=4)

    # Assert
    assert forged.vector_size == 4
    assert forged.aggregator.linear.out_features == 4
    assert forged.aggregator.linear.in_features == 16


def test_forge_megaloc_svd_truncated_leaves_source_unmutated() -> None:
    # Arrange
    model = _build_megaloc(descriptor_dim=12)

    # Act
    forged = forge_megaloc_svd_truncated(model, k=4)

    # Assert
    assert model.vector_size == 12
    assert model.aggregator.linear.out_features == 12
    assert forged is not model
    assert forged.aggregator is not model.aggregator
    assert forged.aggregator.linear is not model.aggregator.linear


def test_forge_megaloc_svd_truncated_reuses_backbone_and_salad() -> None:
    # Arrange
    model = _build_megaloc(descriptor_dim=12)

    # Act
    forged = forge_megaloc_svd_truncated(model, k=4)

    # Assert
    assert forged.backbone is model.backbone
    assert forged.aggregator.salad is model.aggregator.salad


def test_forge_megaloc_svd_truncated_full_rank_keeps_descriptor_dim() -> None:
    # Arrange
    model = _build_megaloc(descriptor_dim=12)

    # Act
    forged = forge_megaloc_svd_truncated(model, k=12)

    # Assert
    assert forged.vector_size == 12
