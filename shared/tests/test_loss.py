"""Tests for loss function and class imbalance handling.

Task 24 — Implement the loss function and class-imbalance handling.
"""

import torch
import torch.nn as nn

from medshield.model import get_loss_function


def test_loss_function_returns_scalar():
    """Test that the loss function returns a scalar for a batch."""
    loss_fn = get_loss_function()

    # Dummy batch: 4 samples, 2 classes
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 0, 1])

    loss = loss_fn(logits, labels)

    # Check that it returns a scalar (0-dimensional tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_class_weights_handle_imbalance():
    """Test that class weights correctly address the imbalance."""
    # Simulate an imbalance: Class 0 has 100 samples, Class 1 has 10 samples
    class_counts = {0: 100, 1: 10}
    loss_fn = get_loss_function(class_counts)

    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert loss_fn.weight is not None

    loss_fn_none = nn.CrossEntropyLoss(weight=loss_fn.weight, reduction="none")

    # Weight formula: total_samples / (num_classes * count)
    # total = 110, num_classes = 2
    # w0 = 110 / (2 * 100) = 0.55
    # w1 = 110 / (2 * 10) = 5.5
    expected_w0 = 110 / 200
    expected_w1 = 110 / 20

    weights = loss_fn.weight.tolist()
    assert abs(weights[0] - expected_w0) < 1e-4
    assert abs(weights[1] - expected_w1) < 1e-4

    # Verify that a misclassification on the minority class yields higher loss
    logits_minority_wrong = torch.tensor([[2.0, -2.0]])  # Predicts 0
    label_minority = torch.tensor([1])
    loss_minority = loss_fn_none(logits_minority_wrong, label_minority)

    logits_majority_wrong = torch.tensor([[-2.0, 2.0]])  # Predicts 1
    label_majority = torch.tensor([0])
    loss_majority = loss_fn_none(logits_majority_wrong, label_majority)

    # Since class 1 has a higher weight, its loss should be proportionally higher
    # Wait, the cross-entropy math: L = -weight[label] * log(softmax(logits)[label])
    # The log(softmax) part is the same in both cases, so the loss scales linearly with the weight
    assert loss_minority.item() > loss_majority.item() * 5.0


def test_loss_decreases_intentional_overfit():
    """Test that the loss decreases on an intentional overfit test over a tiny batch."""
    torch.manual_seed(42)

    # Create a simple linear model to act as a stand-in for our network
    # We just want to prove the loss function works correctly in an optimization loop
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Tiny batch: 2 samples, 10 features
    features = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    loss_fn = get_loss_function(class_counts={0: 10, 1: 10})

    # Initial loss
    logits_init = model(features)
    initial_loss = loss_fn(logits_init, labels).item()

    # Train for a few iterations
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    # Final loss
    logits_final = model(features)
    final_loss = loss_fn(logits_final, labels).item()

    # The loss should decrease significantly
    assert final_loss < initial_loss
    assert final_loss < 0.1  # Should be very close to 0 on a tiny batch
