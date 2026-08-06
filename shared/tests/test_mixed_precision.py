"""Tests for mixed precision training functionality.

Task 30 - Handle mixed precision and device placement.
"""

import logging
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from medshield.model import ModelConfig, train_model


def test_mixed_precision_gracefully_disabled_on_cpu(caplog):
    """Test that mixed precision is gracefully disabled on CPU."""
    caplog.set_level(logging.INFO)

    config = ModelConfig(mixed_precision=True)

    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.randn(8, 10), torch.randint(0, 2, (8,)))
    loader = DataLoader(dataset, batch_size=4)

    device = torch.device("cpu")

    # Train for 1 epoch
    trained_model = train_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=1,
        device=device,
        config=config,
    )

    assert "Mixed precision training (AMP) is DISABLED." in caplog.text


@patch("medshield.model.train.torch.cuda.amp.GradScaler")
@patch("medshield.model.train.torch.autocast")
def test_mixed_precision_enabled_on_cuda(mock_autocast, mock_scaler):
    """Test that mixed precision uses scaler and autocast when on CUDA."""
    config = ModelConfig(mixed_precision=True)

    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.randn(8, 10), torch.randint(0, 2, (8,)))
    loader = DataLoader(dataset, batch_size=4)

    # Create a mock CUDA device
    mock_device = MagicMock()
    mock_device.type = "cuda"

    # Set up mock scaler methods
    mock_scaler_instance = mock_scaler.return_value
    mock_scaler_instance.scale.return_value = MagicMock()

    # Create an autocast context manager mock
    mock_autocast.return_value.__enter__.return_value = None
    mock_autocast.return_value.__exit__.return_value = None

    original_tensor_to = torch.Tensor.to

    def mock_to(self, *args, **kwargs):
        if isinstance(self, MagicMock):
            return self

        new_args = list(args)
        if len(new_args) > 0 and hasattr(new_args[0], "type") and new_args[0].type == "cuda":
            new_args[0] = "cpu"

        if (
            "device" in kwargs
            and hasattr(kwargs["device"], "type")
            and kwargs["device"].type == "cuda"
        ):
            kwargs["device"] = "cpu"

        return original_tensor_to(self, *new_args, **kwargs)

    # Mock model.to and tensor.to to prevent errors when doing pseudo-CUDA
    with patch.object(nn.Linear, "to", return_value=model), patch.object(
        torch.Tensor, "to", new=mock_to
    ):
        train_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=1,
            device=mock_device,
            config=config,
        )

    # Verify GradScaler was instantiated and used
    mock_scaler.assert_called_once()
    assert mock_scaler_instance.scale.call_count == 2  # 2 batches
    assert mock_scaler_instance.step.call_count == 2
    assert mock_scaler_instance.update.call_count == 2

    # Verify autocast was used
    mock_autocast.assert_called_with(device_type="cuda", enabled=True)
