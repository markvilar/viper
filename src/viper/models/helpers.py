"""
Module with helper functionality for image embedding.
"""

import torch
import torchvision.transforms as tfm


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
