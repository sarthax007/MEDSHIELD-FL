"""Tests for selective (critical-parameter) serialization.

Task 33 — Verify that critical parameters are correctly identified,
extracted, round-tripped, and that non-critical parameters remain unchanged.
"""

from __future__ import annotations

import torch

from medshield.model.serialization import (
    CRITICAL_KEY_PATTERNS,
    critical_vector_to_model,
    critical_vector_to_state_dict,
    get_critical_ratio,
    is_critical_parameter,
    model_to_critical_vector,
    state_dict_to_critical_vector,
)
from medshield.model.vit import TumorClassifier


# ---------------------------------------------------------------------------
# is_critical_parameter
# ---------------------------------------------------------------------------


def test_is_critical_parameter_matches_cls_token() -> None:
    """Keys containing 'cls_token' are critical."""
    assert is_critical_parameter("backbone.cls_token")
    assert is_critical_parameter("cls_token")


def test_is_critical_parameter_matches_head() -> None:
    """Keys containing 'head' are critical."""
    assert is_critical_parameter("head.weight")
    assert is_critical_parameter("head.bias")


def test_is_critical_parameter_rejects_non_critical() -> None:
    """Typical backbone keys are NOT critical."""
    assert not is_critical_parameter("backbone.blocks.0.attn.qkv.weight")
    assert not is_critical_parameter("backbone.patch_embed.proj.weight")
    assert not is_critical_parameter("backbone.norm.weight")


# ---------------------------------------------------------------------------
# Critical vector extraction and round-trip
# ---------------------------------------------------------------------------


def test_critical_vector_size_matches_expected() -> None:
    """The critical vector should contain exactly the cls_token + head elements."""
    model = TumorClassifier(num_classes=2, pretrained=False)
    sd = model.state_dict()

    # Manually count expected critical elements
    expected_numel = sum(
        v.numel() for k, v in sd.items() if is_critical_parameter(k)
    )

    vec = model_to_critical_vector(model)
    assert vec.ndim == 1
    assert vec.numel() == expected_numel
    # Sanity: we know for ViT-Base/16 + 2-class head it should be 2306
    assert expected_numel == 768 + 1536 + 2  # cls_token + head.weight + head.bias


def test_critical_round_trip_preserves_critical_params() -> None:
    """Extracting and reinserting the critical vector leaves them identical."""
    model = TumorClassifier(num_classes=2, pretrained=False)
    model.eval()

    # Save original critical params
    orig_sd = {k: v.clone() for k, v in model.state_dict().items() if is_critical_parameter(k)}

    # Extract critical vector
    vec = model_to_critical_vector(model)

    # Create a second model with different weights
    other = TumorClassifier(num_classes=2, pretrained=False)
    other.eval()

    # Load the critical vector into the second model
    critical_vector_to_model(vec, other)

    # Check that critical params now match the original
    other_sd = other.state_dict()
    for key, orig_val in orig_sd.items():
        assert torch.allclose(other_sd[key], orig_val, atol=1e-7), (
            f"Critical parameter '{key}' was not restored correctly."
        )


def test_critical_round_trip_leaves_non_critical_unchanged() -> None:
    """Inserting a critical vector must not touch non-critical parameters."""
    model = TumorClassifier(num_classes=2, pretrained=False)
    model.eval()

    # Save the non-critical params before modification
    orig_non_critical = {
        k: v.clone() for k, v in model.state_dict().items() if not is_critical_parameter(k)
    }

    # Create a dummy critical vector (zeros — the actual values don't matter here)
    vec = model_to_critical_vector(model)
    zeros_vec = torch.zeros_like(vec)

    # Merge zeros into the critical slots
    merged = critical_vector_to_state_dict(zeros_vec, model.state_dict())

    # Non-critical params must be identical
    for key, orig_val in orig_non_critical.items():
        assert torch.equal(merged[key], orig_val), (
            f"Non-critical parameter '{key}' was unexpectedly modified."
        )


def test_critical_round_trip_predictions() -> None:
    """After a critical round-trip the model should produce the same output."""
    model = TumorClassifier(num_classes=2, pretrained=False)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        orig_out = model(dummy)

    vec = model_to_critical_vector(model)

    # Create fresh model, load only critical params from first model
    other = TumorClassifier(num_classes=2, pretrained=False)
    other.eval()

    # Also copy full non-critical state so predictions match
    full_sd = model.state_dict()
    other.load_state_dict(full_sd)

    # Now overwrite critical params using the critical vector API
    critical_vector_to_model(vec, other)

    with torch.no_grad():
        other_out = other(dummy)

    assert torch.allclose(orig_out, other_out, atol=1e-5)


# ---------------------------------------------------------------------------
# get_critical_ratio
# ---------------------------------------------------------------------------


def test_get_critical_ratio_range() -> None:
    """The critical ratio must be a small positive fraction."""
    model = TumorClassifier(num_classes=2, pretrained=False)
    ratio = get_critical_ratio(model)

    assert 0.0 < ratio < 0.01  # well under 1 %
    # For ViT-Base/16 + 2-class head: 2306 / 85_800_194 ≈ 2.69e-05
    assert abs(ratio - 2306 / 85_800_194) < 1e-7
