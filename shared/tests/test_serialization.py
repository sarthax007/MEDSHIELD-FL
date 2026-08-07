"""Tests for model weights serialization and deserialization.

Verifies that model parameters can be flattened into a 1-D vector and restored
exactly, ensuring deterministic ordering, preservation of predictions, and
accurate layout offsets.
"""

from __future__ import annotations

import pytest
import torch

from medshield.model.serialization import (
    get_serialization_layout,
    model_to_vector,
    state_dict_to_vector,
    vector_to_model,
    vector_to_state_dict,
)
from medshield.model.vit import TumorClassifier


def test_serialization_round_trip() -> None:
    """Test that model serialization -> deserialization is a perfect identity map."""
    # Initialize classifier (small classes count, random weights)
    model = TumorClassifier(num_classes=2, pretrained=False)
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(4, 3, 224, 224)

    # Compute baseline predictions
    with torch.no_grad():
        orig_outputs = model(dummy_input)

    # Flatten weights to vector
    vector = model_to_vector(model)
    assert vector.ndim == 1
    assert vector.numel() > 0

    # Instantiate a second model with random weights
    other_model = TumorClassifier(num_classes=2, pretrained=False)
    other_model.eval()

    # Ensure other model makes different predictions initially
    with torch.no_grad():
        other_outputs_before = other_model(dummy_input)
    assert not torch.allclose(orig_outputs, other_outputs_before, atol=1e-4)

    # Load vector into the other model
    vector_to_model(vector, other_model)

    # Compute predictions after restoration
    with torch.no_grad():
        other_outputs_after = other_model(dummy_input)

    # Verify restoration matches the original predictions exactly
    assert torch.allclose(orig_outputs, other_outputs_after, atol=1e-5)


def test_deterministic_key_ordering() -> None:
    """Test that state_dict keys are sorted alphabetically before flattening."""
    state_dict = {
        "z_bias": torch.tensor([10.0]),
        "a_weight": torch.tensor([1.0, 2.0]),
        "m_layer": torch.tensor([5.0, 6.0, 7.0]),
    }

    # Deterministic alphabetical ordering: a_weight, m_layer, z_bias
    expected_values = [1.0, 2.0, 5.0, 6.0, 7.0, 10.0]
    expected_tensor = torch.tensor(expected_values)

    vector = state_dict_to_vector(state_dict)
    assert torch.allclose(vector, expected_tensor)

    # Round trip verification
    reconstructed = vector_to_state_dict(vector, state_dict)
    assert sorted(reconstructed.keys()) == sorted(state_dict.keys())
    for k, v in state_dict.items():
        assert torch.allclose(reconstructed[k], v)


def test_serialization_layout() -> None:
    """Test that get_serialization_layout produces correct offset ranges."""
    state_dict = {
        "layer_b": torch.zeros((2, 2)),
        "layer_a": torch.ones((3,)),
    }

    # Expected sorted order: layer_a (size 3), layer_b (size 4)
    layout = get_serialization_layout(state_dict)

    assert len(layout) == 2

    assert layout[0]["name"] == "layer_a"
    assert layout[0]["shape"] == [3]
    assert layout[0]["numel"] == 3
    assert layout[0]["start_offset"] == 0
    assert layout[0]["end_offset"] == 3

    assert layout[1]["name"] == "layer_b"
    assert layout[1]["shape"] == [2, 2]
    assert layout[1]["numel"] == 4
    assert layout[1]["start_offset"] == 3
    assert layout[1]["end_offset"] == 7
