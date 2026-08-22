"""
Module with helper functionality for image embedding.
"""

from collections.abc import Mapping

import torch
import torchvision.transforms as tfm


def extract_submodule_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """
    Extracts the entries of a state dict belonging to a submodule, with the prefix
    stripped so the result can be loaded into that submodule directly.

    Arguments:
        state_dict - mapping of parameter name to tensor
        prefix - submodule key prefix to select and strip, e.g. "backbone."
    Returns:
        dict mapping the stripped parameter name to tensor
    """
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def convert_grayscale_batch_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """
    Converts an image batch from grayscale to RGB.

    Arguments:
        images - torch.Tensor with shape Bx1xHxW
    Returns:
        torch.Tensor with shape Bx3xHxW
    """
    assert images.dim() == 4, f"invalid batch dimensions: {images.dim()}"
    assert images.shape[1] == 1, f"invalid batch channels: {images.shape[1]}"
    images: torch.Tensor = images.repeat(1, 3, 1, 1)
    return images


def calculate_image_size_dinov2(images: torch.Tensor) -> tuple[int, int]:
    """
    Calculate the largest valid image size for a batch of images. Models with DINOv2
    backbone only accept images with width and height of multiples of 14.

    Arguments:
        images - torch.Tensor of shape BxCxHxW
    Returns:
        tuple - (H, W) with valid height and width
    """
    B, C, H, W = images.shape
    desired_height: int = (H // 14) * 14
    desired_width: int = (W // 14) * 14
    return (desired_height, desired_width)


def resize_image_batch(
    images: torch.Tensor, desired_size: tuple[int, int]
) -> torch.Tensor:
    """
    Resizes an image batch to the desired size.

    Arguments:
        images - torch.Tensor of shape BxCxHxW
        desired_size - tuple of int (H*, W*)
    Returns:
        torch.Tensor of shape BxCxH*xW*
    """
    images_resized: torch.Tensor = tfm.functional.resize(
        images, desired_size, antialias=True
    )
    return images_resized


def resize_to_patch_multiple(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Resizes an image batch so that height and width are multiples of patch_size.

    Arguments:
        images - torch.Tensor of shape BxCxHxW
        patch_size - int, the patch size to align to
    Returns:
        torch.Tensor of shape BxCxH'xW' where H', W' are multiples of patch_size
    """
    B, C, H, W = images.shape
    desired_h: int = (H // patch_size) * patch_size
    desired_w: int = (W // patch_size) * patch_size
    if desired_h == H and desired_w == W:
        return images
    return tfm.functional.resize(images, (desired_h, desired_w), antialias=True)
