"""Tests for the ViT backbone.

Task 21 — Load the ViT-Base/16 backbone.
"""

import torch

from medshield.model import create_vit_backbone


def test_vit_backbone_shape_and_params() -> None:
    """Test that the ViT backbone accepts a 224x224 input and returns (batch_size, 768)."""
    # Use pretrained=False for faster testing without downloading weights
    model = create_vit_backbone(pretrained=False)
    model.eval()

    # Create dummy batch: (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape is (batch_size, 768)
    assert output.shape == (2, 768), f"Expected (2, 768), got {output.shape}"

    # Assert parameter count is around 86M for ViT-Base
    total_params = sum(p.numel() for p in model.parameters())
    # Should be around ~85.8M parameters without head
    assert 85_000_000 < total_params < 87_000_000, f"Unexpected param count: {total_params}"


def test_tumor_classifier_forward() -> None:
    """Test that the TumorClassifier returns properly shaped logits and valid probabilities."""
    from medshield.model import TumorClassifier

    num_classes = 3
    model = TumorClassifier(num_classes=num_classes, pretrained=False)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, num_classes), f"Expected (2, {num_classes}), got {logits.shape}"

    # Check probability distribution
    probs = torch.softmax(logits, dim=-1)

    # Probabilities should sum to 1.0 along the class dimension
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums)), "Softmax probabilities do not sum to 1"
