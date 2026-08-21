"""
Tests for SVD-based truncation of linear projections, verifying output shapes, the
identity-equivalent full-rank case, bias handling, dtype/device preservation, and
input validation.
"""

import pytest
import torch
import torch.nn as nn

from viper.model_decomposition import truncate_linear, truncate_linear_svd


def test_truncate_linear_svd_shapes() -> None:
    # Arrange
    weight = torch.randn(8, 5)
    bias = torch.randn(8)

    # Act
    new_weight, new_bias = truncate_linear_svd(weight, bias, k=3)

    # Assert
    assert new_weight.shape == (3, 5)
    assert new_bias is not None
    assert new_bias.shape == (3,)


def test_truncate_linear_svd_none_bias_passes_through() -> None:
    # Arrange
    weight = torch.randn(6, 4)

    # Act
    new_weight, new_bias = truncate_linear_svd(weight, None, k=2)

    # Assert
    assert new_weight.shape == (2, 4)
    assert new_bias is None


def test_truncate_linear_svd_full_rank_is_projection_equivalent() -> None:
    # Arrange
    # k == out reproduces the original mapping up to an orthonormal change of basis
    # of the output; the L2 norm of the output vector is therefore preserved.
    weight = torch.randn(5, 7)
    bias = torch.randn(5)
    x = torch.randn(4, 7)

    # Act
    new_weight, new_bias = truncate_linear_svd(weight, bias, k=5)
    original = x @ weight.T + bias
    truncated = x @ new_weight.T + new_bias

    # Assert
    torch.testing.assert_close(
        original.norm(dim=1), truncated.norm(dim=1), atol=1e-5, rtol=1e-5
    )


def test_truncate_linear_svd_recovers_dominant_directions() -> None:
    # Arrange
    # Build a weight with a known dominant singular direction; truncating to k=1
    # must retain (nearly) all of the projection's energy.
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(6, 6))
    V, _ = torch.linalg.qr(torch.randn(4, 4))
    singular_values = torch.tensor([10.0, 0.1, 0.05, 0.01])
    weight = U[:, :4] @ torch.diag(singular_values) @ V.T

    # Act
    new_weight, _ = truncate_linear_svd(weight, None, k=1)

    # Assert
    retained = new_weight.norm()
    total = weight.norm()
    assert retained / total > 0.99


def test_truncate_linear_svd_promotes_low_precision_and_casts_back() -> None:
    # Arrange
    weight = torch.randn(8, 5, dtype=torch.float16)
    bias = torch.randn(8, dtype=torch.float16)

    # Act
    new_weight, new_bias = truncate_linear_svd(weight, bias, k=3)

    # Assert
    assert new_weight.dtype == torch.float16
    assert new_bias is not None
    assert new_bias.dtype == torch.float16


@pytest.mark.parametrize("k", [0, -1, 9])
def test_truncate_linear_svd_rejects_invalid_k(k: int) -> None:
    # Arrange
    weight = torch.randn(8, 5)

    # Act / Assert
    with pytest.raises(ValueError):
        truncate_linear_svd(weight, None, k=k)


def test_truncate_linear_svd_rejects_non_2d_weight() -> None:
    # Arrange
    weight = torch.randn(8)

    # Act / Assert
    with pytest.raises(ValueError):
        truncate_linear_svd(weight, None, k=1)


def test_truncate_linear_svd_rejects_mismatched_bias() -> None:
    # Arrange
    weight = torch.randn(8, 5)
    bias = torch.randn(7)

    # Act / Assert
    with pytest.raises(ValueError):
        truncate_linear_svd(weight, bias, k=3)


def test_truncate_linear_returns_layer_with_target_dimensions() -> None:
    # Arrange
    linear = nn.Linear(10, 6)

    # Act
    truncated = truncate_linear(linear, k=4)

    # Assert
    assert isinstance(truncated, nn.Linear)
    assert truncated.in_features == 10
    assert truncated.out_features == 4
    assert truncated.bias is not None


def test_truncate_linear_without_bias() -> None:
    # Arrange
    linear = nn.Linear(10, 6, bias=False)

    # Act
    truncated = truncate_linear(linear, k=4)

    # Assert
    assert truncated.bias is None
    assert truncated.weight.shape == (4, 10)


def test_truncate_linear_preserves_output_norm_at_full_rank() -> None:
    # Arrange
    linear = nn.Linear(7, 5)
    x = torch.randn(3, 7)

    # Act
    truncated = truncate_linear(linear, k=5)

    # Assert
    with torch.no_grad():
        original = linear(x)
        result = truncated(x)
    torch.testing.assert_close(
        original.norm(dim=1), result.norm(dim=1), atol=1e-5, rtol=1e-5
    )
