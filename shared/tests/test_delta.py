"""Tests for model update delta computation and application.

Task 34 — Extract model update deltas.
Verifies that delta = local - global, and global + delta = local.
"""

import copy

import pytest
import torch

from medshield.model.delta import (
    apply_delta,
    apply_model_delta,
    compute_delta,
    compute_model_delta,
)
from medshield.model.serialization import (
    model_to_critical_vector,
    model_to_vector,
)
from medshield.model.vit import TumorClassifier


def test_compute_apply_delta_basic() -> None:
    """Test delta calculation and application on simple tensors."""
    global_vec = torch.tensor([1.0, 2.0, 3.0])
    local_vec = torch.tensor([1.5, 1.5, 4.0])

    delta = compute_delta(global_vec, local_vec)
    expected_delta = torch.tensor([0.5, -0.5, 1.0])
    assert torch.allclose(delta, expected_delta)

    reconstructed_local = apply_delta(global_vec, delta)
    assert torch.allclose(reconstructed_local, local_vec)


def test_compute_apply_delta_shape_mismatch() -> None:
    """Test that shape mismatches raise ValueError."""
    global_vec = torch.tensor([1.0, 2.0])
    local_vec = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_delta(global_vec, local_vec)

    with pytest.raises(ValueError, match="Shape mismatch"):
        apply_delta(global_vec, local_vec)


def test_model_delta_full() -> None:
    """Test full model delta computation and application."""
    global_model = TumorClassifier(num_classes=2, pretrained=False)
    local_model = TumorClassifier(num_classes=2, pretrained=False)

    # Compute delta
    delta = compute_model_delta(global_model, local_model, critical_only=False)

    # Apply delta to a copy of the global model
    reconstructed_model = copy.deepcopy(global_model)
    apply_model_delta(reconstructed_model, delta, critical_only=False)

    # Check if reconstructed model equals local model
    local_vec = model_to_vector(local_model)
    reconstructed_vec = model_to_vector(reconstructed_model)
    assert torch.allclose(local_vec, reconstructed_vec, atol=1e-5)


def test_model_delta_critical_only() -> None:
    """Test critical-only model delta computation and application."""
    global_model = TumorClassifier(num_classes=2, pretrained=False)
    local_model = TumorClassifier(num_classes=2, pretrained=False)

    # Compute critical delta
    delta = compute_model_delta(global_model, local_model, critical_only=True)

    # Apply delta to a copy of the global model
    reconstructed_model = copy.deepcopy(global_model)
    apply_model_delta(reconstructed_model, delta, critical_only=True)

    # Check if critical parameters equal local model's critical parameters
    local_critical_vec = model_to_critical_vector(local_model)
    reconstructed_critical_vec = model_to_critical_vector(reconstructed_model)
    assert torch.allclose(local_critical_vec, reconstructed_critical_vec, atol=1e-5)

    # Check if non-critical parameters are still equal to the original global model
    global_full_vec = model_to_vector(global_model)
    reconstructed_full_vec = model_to_vector(reconstructed_model)

    # We can't just check the whole vector equality because critical params changed.
    # But we know delta was only applied to critical params.
    # To check non-critical, we can apply the inverse delta and check equality to global.
    inverse_delta = -delta
    apply_model_delta(reconstructed_model, inverse_delta, critical_only=True)

    reverted_vec = model_to_vector(reconstructed_model)
    assert torch.allclose(global_full_vec, reverted_vec, atol=1e-5)
