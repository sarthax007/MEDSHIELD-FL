"""Tests for medshield.data.preprocess — Task 14 acceptance criteria."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medshield.data.preprocess import normalize_intensity, preprocess_slice, resize_slice


def test_normalize_intensity() -> None:
    # Create a tensor with known mean and std
    tensor = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ]
    )

    norm = normalize_intensity(tensor)

    # Channel 0: mean=2.5, std=~1.29 (population std is 1.118, sample std is 1.29)
    # Actually, PyTorch std(unbiased=False) uses population std -> 1.118
    assert torch.allclose(norm[0].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(norm[0].std(unbiased=False), torch.tensor(1.0), atol=1e-5)

    # Channel 1: all 5.0, mean=5.0, std=0.0
    # Normalization should return close to 0 due to eps
    assert torch.allclose(norm[1], torch.tensor(0.0), atol=1e-5)


def test_resize_slice() -> None:
    # Create a 4x20x20 tensor
    tensor = torch.ones((4, 20, 20))
    resized = resize_slice(tensor, (224, 224))

    assert resized.shape == (4, 224, 224)
    assert torch.allclose(resized, torch.tensor(1.0))


def test_preprocess_slice_from_numpy() -> None:
    # 4 channels, 240x240
    data = np.random.rand(4, 240, 240).astype(np.float32)

    # Convert to 3 channels, 224x224
    tensor = preprocess_slice(data, out_channels=3, target_size=(224, 224))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 224, 224)

    # Verify normalization worked (mean ~ 0, std ~ 1)
    for c in range(3):
        assert torch.allclose(tensor[c].mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(tensor[c].std(unbiased=False), torch.tensor(1.0), atol=1e-2)


def test_preprocess_slice_4_channels() -> None:
    data = np.random.rand(4, 100, 100)
    tensor = preprocess_slice(data, out_channels=4, target_size=(256, 256))

    assert tensor.shape == (4, 256, 256)


def test_preprocess_slice_invalid_channels() -> None:
    data = np.ones((4, 10, 10))
    with pytest.raises(ValueError, match="out_channels must be 3 or 4"):
        preprocess_slice(data, out_channels=2)


def test_preprocess_slice_deterministic() -> None:
    # Set a fixed seed so we have repeatable random data
    rng = np.random.default_rng(42)
    data = rng.random((4, 240, 240))

    tensor1 = preprocess_slice(data)
    tensor2 = preprocess_slice(data)

    assert torch.equal(tensor1, tensor2)
