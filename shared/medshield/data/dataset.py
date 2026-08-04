"""PyTorch Dataset and DataLoader for 2D MRI slices.

Task 18 — Build PyTorch Dataset and DataLoader classes.

Wraps the preprocessed and partitioned slices in a PyTorch Dataset,
applying label mapping and optional data augmentation on the fly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from medshield.data.augmentation import get_eval_transforms, get_train_transforms
from medshield.data.labels import map_raw_label
from medshield.data.preprocess import preprocess_slice

logger = logging.getLogger(__name__)


class BraTS2DSliceDataset(Dataset):
    """PyTorch Dataset for BraTS 2D slices.

    Loads `.npy` slice files on the fly, preprocesses them, maps labels,
    and applies optional transformations.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_dir: Path | str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        out_channels: int = 4,
        target_size: tuple[int, int] = (224, 224),
    ) -> None:
        """
        Parameters
        ----------
        manifest_df : pd.DataFrame
            DataFrame containing 'filename' and 'has_tumor' columns.
        data_dir : Path | str
            Directory containing the `.npy` slice files.
        transform : Callable | None
            Optional PyTorch transformation pipeline (from augmentation.py).
        out_channels : int
            Number of output channels (3 or 4).
        target_size : tuple[int, int]
            Spatial size for the ViT input (default 224x224).
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.out_channels = out_channels
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest_df.iloc[idx]

        # 1. Load slice
        file_path = self.data_dir / row["filename"]
        data_array = np.load(file_path)

        # 2. Preprocess slice
        img_tensor = preprocess_slice(
            data_array, out_channels=self.out_channels, target_size=self.target_size
        )

        # 3. Apply augmentation if any
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # 4. Map label
        # We expect a string like "tumor" or boolean True/False in "has_tumor" or another label column
        # In slicer.py, we have "has_tumor" as boolean.
        raw_label = row["has_tumor"]
        label_idx = map_raw_label(raw_label)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return img_tensor, label_tensor


def create_dataloaders(
    manifest_df: pd.DataFrame,
    data_dir: Path | str,
    batch_size: int = 32,
    is_train: bool = True,
    num_workers: int = 4,
    transform_config: dict | None = None,
    out_channels: int = 4,
    target_size: tuple[int, int] = (224, 224),
) -> DataLoader:
    """Factory function to build a DataLoader from a slice manifest.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        The partition manifest defining the split.
    data_dir : Path | str
        Directory containing the `.npy` files.
    batch_size : int
        Number of samples per batch (default 32).
    is_train : bool
        If True, applies training augmentations and shuffles data.
        If False, applies no augmentations and doesn't shuffle.
    num_workers : int
        Number of subprocesses for data loading.
    transform_config : dict | None
        Configuration passed to augmentation pipeline.
    out_channels : int
        Number of channels (3 or 4).
    target_size : tuple[int, int]
        Spatial dimensions (H, W).

    Returns
    -------
    DataLoader
        The PyTorch DataLoader ready for iteration.
    """
    if is_train:
        transform = get_train_transforms(transform_config)
        shuffle = True
    else:
        transform = get_eval_transforms()
        shuffle = False

    dataset = BraTS2DSliceDataset(
        manifest_df=manifest_df,
        data_dir=data_dir,
        transform=transform,
        out_channels=out_channels,
        target_size=target_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
