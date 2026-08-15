"""Data augmentation pipeline for medical MRI slices.

Task 16 — Implement data augmentation.

Provides medically valid training-time augmentations for 2D MRI slices
and utilities to configure, apply, and visualize them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

logger = logging.getLogger(__name__)


def get_train_transforms(config: dict | None = None) -> transforms.Compose:
    """Create the data augmentation pipeline for the training split.

    Uses spatial transforms that are medically valid for axial brain MRI slices.

    Parameters
    ----------
    config : dict | None
        Optional configuration for the transforms (e.g., degrees, translation bounds).

    Returns
    -------
    transforms.Compose
        A torchvision v2 transform pipeline.
    """
    if config is None:
        config = {}

    degrees = config.get("rotation_degrees", 10.0)
    translate = config.get("translation_bounds", (0.05, 0.05))
    flip_p = config.get("horizontal_flip_p", 0.5)

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Create the identity pipeline for evaluation (no augmentation).

    Returns
    -------
    transforms.Compose
        An empty transform pipeline.
    """
    return transforms.Compose([transforms.Identity()])


def visualize_augmentations(
    tensor: torch.Tensor,
    transform: transforms.Compose,
    output_path: Path,
    num_samples: int = 5,
) -> None:
    """Apply augmentations multiple times to a single tensor and save a visual grid.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor of shape ``(C, H, W)``. Should be preprocessed/normalized.
    transform : transforms.Compose
        The augmentation pipeline to apply.
    output_path : Path
        File path to save the generated PNG image.
    num_samples : int
        Number of augmented variations to generate (including the original).
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(3 * (num_samples + 1), 3))

    # We assume tensor is in (C, H, W) and normalized. For visualization,
    # we take the first channel if there's only 1, or construct an RGB if >= 3.
    # To keep it simple, we'll just show the middle channel or an average if multiple.

    def tensor_to_img(t: torch.Tensor) -> np.ndarray:
        data = t.cpu().numpy()
        # Create a single-channel image by averaging across channels
        img = data.mean(axis=0)
        # Normalize to [0, 1] for display
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return np.asarray(img)

    # Original
    axes[0].imshow(tensor_to_img(tensor), cmap="gray", origin="lower")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Augmented
    for i in range(1, num_samples + 1):
        aug_tensor = transform(tensor)
        axes[i].imshow(tensor_to_img(aug_tensor), cmap="gray", origin="lower")
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved augmentation visualization to %s", output_path)
