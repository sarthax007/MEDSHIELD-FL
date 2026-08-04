"""Tests for model configuration and registry.

Task 23 — Implement a model configuration and registry.
"""

import json

from medshield.model import ModelConfig, TumorClassifier, create_model


def test_model_config_serialization(tmp_path):
    """Test that a ModelConfig can be correctly serialized and deserialized."""
    config = ModelConfig(
        model_name="vit_base_patch16_224", num_classes=4, pretrained=False, drop_rate=0.1
    )

    # Save to temp file
    config_path = tmp_path / "model_config.json"
    config.save(config_path)

    # Check the file contents directly
    with open(config_path) as f:
        data = json.load(f)

    assert data["num_classes"] == 4
    assert data["pretrained"] is False
    assert data["drop_rate"] == 0.1

    # Reload from file
    loaded_config = ModelConfig.load(config_path)
    assert loaded_config == config, "Deserialized config does not match original"


def test_create_model_consistency():
    """Test that two builds from the same config produce identical architectures."""
    config = ModelConfig(
        model_name="vit_base_patch16_224", num_classes=3, pretrained=False, drop_rate=0.2
    )

    model1 = create_model(config)
    model2 = create_model(config)

    assert isinstance(model1, TumorClassifier)
    assert isinstance(model2, TumorClassifier)

    # Verify both models have the same head structure
    assert model1.head.out_features == 3
    assert model2.head.out_features == 3

    # Count parameters
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    assert (
        params1 == params2
    ), "Models built from the same config should have identical parameter counts"
