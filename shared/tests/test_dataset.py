"""Tests for the Dataset and DataLoader classes.

Task 18 — Build PyTorch Dataset and DataLoader classes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from medshield.data.dataset import BraTS2DSliceDataset, create_dataloaders


@pytest.fixture
def dummy_data_dir(tmp_path: Path) -> Path:
    """Fixture to create a temporary directory with dummy `.npy` slice files."""
    data_dir = tmp_path / "slices"
    data_dir.mkdir()

    # Create 5 dummy slices
    # The slicer creates slices with shape (4, H, W) for 4 modalities
    for i in range(5):
        dummy_slice = np.random.rand(4, 240, 240).astype(np.float32)
        np.save(data_dir / f"slice_{i}.npy", dummy_slice)

    return data_dir


@pytest.fixture
def dummy_manifest(dummy_data_dir: Path) -> pd.DataFrame:
    """Fixture to create a dummy manifest DataFrame."""
    return pd.DataFrame(
        {
            "filename": [f"slice_{i}.npy" for i in range(5)],
            "has_tumor": [True, False, True, False, True],
        }
    )


def test_dataset_returns_correct_shapes(dummy_manifest: pd.DataFrame, dummy_data_dir: Path) -> None:
    """Test that the dataset returns tensors of expected shapes."""
    dataset = BraTS2DSliceDataset(
        manifest_df=dummy_manifest, data_dir=dummy_data_dir, out_channels=3, target_size=(224, 224)
    )

    assert len(dataset) == 5

    img, label = dataset[0]

    # Check shapes
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label.shape == torch.Size([])  # Scalar tensor
    assert label.item() == 1  # True maps to 1


def test_dataloader_yields_batches(dummy_manifest: pd.DataFrame, dummy_data_dir: Path) -> None:
    """Test that the dataloader yields correctly shaped batches without crashing."""
    loader = create_dataloaders(
        manifest_df=dummy_manifest,
        data_dir=dummy_data_dir,
        batch_size=2,
        is_train=True,
        num_workers=0,  # 0 for testing in same process
        out_channels=3,
        target_size=(224, 224),
    )

    batches = list(loader)

    # 5 items, batch size 2 -> 3 batches (sizes: 2, 2, 1)
    assert len(batches) == 3

    # Check first batch
    img_batch, label_batch = batches[0]
    assert img_batch.shape == (2, 3, 224, 224)
    assert label_batch.shape == (2,)
