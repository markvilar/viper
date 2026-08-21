"""
SALAD feature aggregation module and helpers.

Adapted from the MegaLoc reference implementation (gmberton/MegaLoc, MIT license),
which in turn adapts Sinkhorn OTP code from OpenGlue (MIT license):
https://github.com/ucuapps/OpenGlue
"""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def log_otp_solver(
    log_a: Tensor,
    log_b: Tensor,
    M: Tensor,
    num_iters: int = 20,
    reg: float = 1.0,
) -> Tensor:
    """Sinkhorn matrix scaling for differentiable optimal transport."""
    M = M / reg
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(
    S: Tensor,
    dustbin_score: float = 1.0,
    num_iters: int = 3,
    reg: float = 1.0,
) -> Tensor:
    """Compute soft assignment probabilities via Sinkhorn OTP."""
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a = norm.expand(m + 1).contiguous()
    log_b = norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class L2Norm(nn.Module):
    """L2-normalises a tensor along a given dimension."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2.0, dim=self.dim)


class SALAD(nn.Module):
    """
    Optimal-transport feature aggregator (SALAD / FeatureAggregator).

    Aggregates spatial patch features and a CLS token into a compact
    global descriptor via differentiable optimal transport (Sinkhorn).

    Arguments:
        num_channels: C, input feature channels from the backbone
        num_clusters: K, number of cluster centers
        cluster_dim: D_k, dimensionality per cluster
        token_dim: D_t, dimensionality of the global token branch
        mlp_dim: hidden width for all internal MLPs
        dropout: dropout probability (0 to disable)
    """

    def __init__(
        self,
        num_channels: int = 768,
        num_clusters: int = 64,
        cluster_dim: int = 256,
        token_dim: int = 256,
        mlp_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        drop: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(num_channels, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, mlp_dim, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(mlp_dim, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, mlp_dim, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(mlp_dim, num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def output_dim(self) -> int:
        return self.num_clusters * self.cluster_dim + self.token_dim

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            x: tuple of (patches (B, C, H, W), cls_token (B, C))
        Returns:
            global descriptor (B, num_clusters * cluster_dim + token_dim)
        """
        patches, cls_token = x
        f = self.cluster_features(patches).flatten(2)
        p = self.score(patches).flatten(2)
        t = self.token_features(cls_token)
        p = get_matching_probs(p, self.dust_bin, num_iters=3)
        p = torch.exp(p)[:, :-1, :]
        descriptor = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),
                F.normalize(torch.einsum("bdn,bkn->bdk", f, p), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        return F.normalize(descriptor, p=2, dim=-1)


def build_salad(dropout: float = 0.0) -> SALAD:
    """
    Construct a SALAD aggregator configured for MegaLoc's DINOv2 ViT-B/14 backbone.

    The output dimension is 64 * 256 + 256 = 16640.
    """
    return SALAD(
        num_channels=768,
        num_clusters=64,
        cluster_dim=256,
        token_dim=256,
        mlp_dim=512,
        dropout=dropout,
    )


def build_salad_from_state_dict(
    state_dict: Mapping[str, Tensor], dropout: float = 0.0
) -> SALAD:
    """
    Construct a SALAD aggregator whose configuration is derived from the weight
    shapes in a state dict, then load those weights.

    All dimensions are inferred from the weights so the factory works for any
    SALAD variant, not just the MegaLoc default:

        num_channels <- token_features.0.weight  (in_features)
        mlp_dim      <- token_features.0.weight  (out_features)
        token_dim    <- token_features.2.weight  (out_features)
        cluster_dim  <- cluster_features.3.weight (out_channels)
        num_clusters <- score.3.weight            (out_channels)

    Arguments:
        state_dict - SALAD state dict (keys relative to the SALAD module)
        dropout - dropout probability for the constructed module
    Returns:
        SALAD module with the given weights loaded (strict)
    """
    token_features_in = state_dict["token_features.0.weight"]
    token_features_out = state_dict["token_features.2.weight"]
    cluster_features_out = state_dict["cluster_features.3.weight"]
    score_out = state_dict["score.3.weight"]

    salad = SALAD(
        num_channels=token_features_in.shape[1],
        num_clusters=score_out.shape[0],
        cluster_dim=cluster_features_out.shape[0],
        token_dim=token_features_out.shape[0],
        mlp_dim=token_features_in.shape[0],
        dropout=dropout,
    )
    salad.load_state_dict(state_dict, strict=True)
    return salad
