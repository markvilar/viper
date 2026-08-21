"""
SVD-based truncation of linear projections.

Generic, model-agnostic primitives for reducing the output dimension of a linear
layer while preserving as much of its representational structure as possible. The
tensor-level function operates on raw weight/bias tensors; the module-level
convenience returns a ready-to-use ``nn.Linear``.
"""

import torch
import torch.nn as nn
from torch import Tensor

# Floating point types promoted to float32 for the SVD to keep the decomposition
# numerically stable, then cast back to the source dtype.
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def truncate_linear_svd(
    weight: Tensor,
    bias: Tensor | None,
    k: int,
) -> tuple[Tensor, Tensor | None]:
    """
    Truncate a linear projection to its k highest-variance output directions via
    truncated SVD.

    Given weight W (shape (out, in)) with W = U Σ Vᵀ, returns the rank-k
    projection into the top-k left-singular (output) subspace:

        W_new = Σ[:k] * Vᵀ[:k, :]     shape (k, in)
        b_new = U[:, :k]ᵀ @ b          shape (k,)

    The result is an orthonormal re-expression of the original descriptor's
    dominant subspace, so after L2 normalisation the truncated projection stays
    geometrically close to the original at initialisation. ``k == out`` returns an
    identity-equivalent projection; a ``None`` bias passes through as ``None``.

    The SVD is computed on the source device, promoting to float32 when the weight
    is lower precision (fp16/bf16) for numerical stability. The returned tensors
    inherit the source weight's device and dtype.

    Arguments:
        weight - projection weight, shape (out, in)
        bias   - projection bias, shape (out,), or None
        k      - target output dimension, 0 < k <= out
    Returns:
        (W_new, b_new) - truncated weight (k, in) and bias (k,) or None
    """
    if weight.dim() != 2:
        raise ValueError(
            f"expected a 2D weight (out, in), got shape {tuple(weight.shape)}"
        )
    out_features, _ = weight.shape
    if not 0 < k <= out_features:
        raise ValueError(f"expected 0 < k <= out_features ({out_features}), got k={k}")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError(
            f"expected bias of shape ({out_features},), got {tuple(bias.shape)}"
        )

    source_dtype = weight.dtype
    compute_dtype = (
        torch.float32 if source_dtype in _LOW_PRECISION_DTYPES else source_dtype
    )
    weight_compute = weight.to(compute_dtype)

    # Economy-size SVD: full_matrices=False avoids materialising the unused
    # (in, in) part of V and is sufficient for the top-k singular vectors.
    U, S, Vh = torch.linalg.svd(weight_compute, full_matrices=False)

    U_k = U[:, :k]
    new_weight = (S[:k].unsqueeze(1) * Vh[:k, :]).to(source_dtype)

    new_bias: Tensor | None = None
    if bias is not None:
        new_bias = (U_k.transpose(0, 1) @ bias.to(compute_dtype)).to(source_dtype)

    return new_weight, new_bias


def truncate_linear(linear: nn.Linear, k: int) -> nn.Linear:
    """
    Return a new nn.Linear(in_features, k) initialised from the truncated SVD of
    ``linear`` (see truncate_linear_svd), mapping into the top-k output subspace of
    the original layer.

    The returned layer inherits the source layer's device and dtype, and has a bias
    if and only if the source layer does.

    Arguments:
        linear - source linear projection
        k      - target output dimension, 0 < k <= linear.out_features
    Returns:
        nn.Linear with in_features == linear.in_features and out_features == k
    """
    weight = linear.weight.detach()
    bias = linear.bias.detach() if linear.bias is not None else None

    new_weight, new_bias = truncate_linear_svd(weight, bias, k)

    truncated = nn.Linear(
        linear.in_features,
        k,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        truncated.weight.copy_(new_weight)
        if new_bias is not None:
            truncated.bias.copy_(new_bias)
    return truncated
