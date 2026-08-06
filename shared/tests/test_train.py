"""Tests for the local training loop.

Task 26 — Build the local training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medshield.model import ModelConfig, get_device, get_optimizer, get_scheduler, train_model


def test_get_device():
    """Test that get_device returns a valid torch.device object."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda"]


def test_training_loop_end_to_end():
    """Test a short run completes end-to-end and produces a trained model object."""
    torch.manual_seed(42)

    # Create a small dummy dataset
    num_samples = 8
    input_size = 10
    num_classes = 2

    features = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(features, labels)

    # Use small batch size to force multiple iterations
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Create dummy model and required components
    model = nn.Linear(input_size, num_classes)

    config = ModelConfig(learning_rate=0.1, epochs=2)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    loss_fn = nn.CrossEntropyLoss()

    # Record initial weights to ensure they change
    initial_weights = model.weight.clone()

    # Run the training loop for 2 epochs
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=2,
        device=torch.device("cpu"),  # Force CPU for consistent testing
    )

    # Assert model was returned and modified
    assert trained_model is model
    assert not torch.allclose(initial_weights, trained_model.weight)
