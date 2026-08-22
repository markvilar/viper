"""
DINOv2 Vision Transformer backbone and sub-modules.

Adapted from the MegaLoc reference implementation (gmberton/MegaLoc, MIT license).
"""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEmbedding(nn.Module):
    """Projects image patches into embedding vectors via a strided convolution."""

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LayerScale(nn.Module):
    """Per-channel learnable scale, as used in CaiT and DINOv2."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Mlp(nn.Module):
    """Feed-forward MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """ViT transformer block with pre-norm and LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2(nn.Module):
    """
    DINOv2 Vision Transformer backbone.

    Returns spatial patch features (B, C, H/p, W/p) and a CLS token (B, C).
    Positional embeddings are interpolated for inputs that differ in size from
    the training resolution.
    """

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = embed_dim

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Masked-image-modeling token from DINOv2 pretraining; carried so trained
        # checkpoints load strictly, but unused during inference feature extraction.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.interpolate_offset = 0.1
        self.interpolate_antialias = False

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        cls_pe = pos_embed[:, 0]
        patch_pe = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        patch_pe = F.interpolate(
            patch_pe.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        if (w0, h0) != patch_pe.shape[-2:]:
            raise RuntimeError(
                f"interpolated positional grid {tuple(patch_pe.shape[-2:])} "
                f"does not match expected {(w0, h0)}"
            )
        patch_pe = patch_pe.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pe.unsqueeze(0), patch_pe), dim=1).to(previous_dtype)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Arguments:
            images: (B, 3, H, W) — H and W must be multiples of patch_size
        Returns:
            patches: (B, C, H/patch_size, W/patch_size)
            cls_token: (B, C)
        """
        B, _, H, W = images.shape
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self._interpolate_pos_encoding(x, H, W)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        patches = patch_tokens.reshape(
            B, H // self.patch_size, W // self.patch_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        return patches, cls_token


def build_dinov2_vitb14() -> DINOv2:
    """Construct a DINOv2 ViT-B/14 backbone (weights not loaded)."""
    return DINOv2(
        image_size=518,
        patch_size=14,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )


def build_dinov2_vitb14_from_state_dict(state_dict: Mapping[str, Tensor]) -> DINOv2:
    """
    Construct a DINOv2 ViT-B/14 backbone and load weights from a state dict.

    The ViT-B/14 architecture is fixed by definition (num_heads in particular is
    not recoverable from any weight shape), so the module is built with the known
    architecture and the weights are loaded strictly — a mismatch raises.

    Arguments:
        state_dict - DINOv2 state dict (keys relative to the DINOv2 module)
    Returns:
        DINOv2 module with the given weights loaded (strict)
    """
    dinov2 = build_dinov2_vitb14()
    dinov2.load_state_dict(state_dict, strict=True)
    return dinov2
