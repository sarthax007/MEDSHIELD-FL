"""Tests for the optimizer and LR scheduler configuration.

Task 25 — Implement the optimizer and learning-rate scheduler.
"""

import torch
import torch.nn as nn

from medshield.model import ModelConfig, get_optimizer, get_scheduler


def test_optimizer_initialization():
    """Test that the optimizer is correctly constructed from the config."""
    model = nn.Linear(10, 2)
    config = ModelConfig(learning_rate=0.01, weight_decay=1e-3)

    optimizer = get_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 1e-3


def test_scheduler_learning_rate_decreases():
    """Test that the learning rate changes over epochs as intended."""
    model = nn.Linear(10, 2)
    config = ModelConfig(learning_rate=0.1, epochs=10)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    initial_lr = optimizer.param_groups[0]["lr"]

    # Step optimizer then scheduler
    optimizer.step()
    scheduler.step()

    next_lr = optimizer.param_groups[0]["lr"]

    # Cosine annealing should decrease the LR
    assert next_lr < initial_lr


def test_optimizer_stable_loss_run():
    """Test a short training loop to show stable (non-diverging) loss."""
    torch.manual_seed(42)
    model = nn.Linear(10, 2)
    config = ModelConfig(learning_rate=0.1, epochs=5)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy data
    features = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 0, 1])

    initial_loss = loss_fn(model(features), labels).item()

    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    final_loss = loss_fn(model(features), labels).item()

    # Check that it didn't diverge to NaN or Infinity, and it decreased
    assert not torch.isnan(torch.tensor(final_loss))
    assert final_loss < initial_loss
