"""Image preprocessing pipeline.

Task 14 — Build the image preprocessing pipeline.

Standardizes extracted 2D MRI slices for Vision Transformer (ViT) input.
Handles tensor conversion, resizing, channel selection, and intensity normalization.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as F_vision  # noqa: N812


def normalize_intensity(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply per-channel Z-score normalization.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor of shape ``(C, H, W)``.
    eps : float
        Small epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Normalized tensor of shape ``(C, H, W)``.
    """
    # Calculate mean and std over spatial dimensions (H, W) per channel (C)
    mean = tensor.mean(dim=(-2, -1), keepdim=True)
    std = tensor.std(dim=(-2, -1), keepdim=True, unbiased=False)

    # If a channel is completely uniform, std will be 0. Use eps to avoid NaN.
    return (tensor - mean) / (std + eps)


def resize_slice(tensor: torch.Tensor, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Resize the spatial dimensions of the tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor of shape ``(C, H, W)``.
    size : tuple[int, int]
        Target spatial size ``(H, W)``.

    Returns
    -------
    torch.Tensor
        Resized tensor of shape ``(C, size[0], size[1])``.
    """
    # F_vision.resize expects [..., H, W]
    return F_vision.resize(
        tensor, list(size), interpolation=F_vision.InterpolationMode.BILINEAR, antialias=True
    )


def preprocess_slice(
    data: np.ndarray | torch.Tensor,
    out_channels: int = 4,
    target_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Full preprocessing pipeline for a 2D MRI slice.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input 2D slice of shape ``(C, H, W)``, typically extracted by ``slicer.py``.
    out_channels : int
        Number of output channels (3 or 4). If 3, drops the first channel (T1).
    target_size : tuple[int, int]
        Spatial size for the ViT input (default 224x224).

    Returns
    -------
    torch.Tensor
        Preprocessed float32 tensor of shape ``(out_channels, target_size[0], target_size[1])``.
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = data.clone()

    tensor = tensor.to(torch.float32)

    if out_channels == 3 and tensor.shape[0] >= 4:
        # Drop the first channel (e.g. T1) and keep the rest (T1ce, T2, FLAIR)
        tensor = tensor[1:4, :, :]
    elif out_channels not in (3, 4):
        raise ValueError(f"out_channels must be 3 or 4, got {out_channels}")

    tensor = resize_slice(tensor, size=target_size)
    tensor = normalize_intensity(tensor)

    return tensor
