"""Tests for model checkpointing.

Task 28 - Implement checkpointing.
"""

import os
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.optim import AdamW

from medshield.model import ModelConfig, load_checkpoint, save_checkpoint


def test_save_load_checkpoint():
    """Test that a checkpoint can be saved and loaded, preserving weights."""
    config = ModelConfig(
        model_name="vit_base_patch16_224",
        num_classes=2,
        pretrained=False,
    )

    # Simple dummy model instead of full ViT for speed
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    with TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_checkpoint.pt")

        # Save a checkpoint
        save_checkpoint(
            model=model,
            epoch=5,
            val_loss=0.1234,
            config=config,
            filepath=filepath,
            optimizer=optimizer,
        )

        assert os.path.exists(filepath)

        # Modify the model weights to prove we load the old ones back
        with torch.no_grad():
            model.weight.fill_(0.0)
            model.bias.fill_(0.0)

        assert (model.weight == 0.0).all()

        # Create a fresh optimizer
        fresh_optimizer = AdamW(model.parameters(), lr=1e-1)

        # Load the checkpoint
        loaded_config, loaded_epoch, loaded_val_loss = load_checkpoint(
            filepath=filepath,
            model=model,
            optimizer=fresh_optimizer,
        )

        # Verify values
        assert loaded_epoch == 5
        assert loaded_val_loss == 0.1234
        assert loaded_config.num_classes == 2

        # Check weights are no longer 0
        assert (model.weight != 0.0).any()

        # Check optimizer state (e.g., learning rate was restored from the old optimizer state)
        # Actually AdamW state_dict keeps lr per param_group
        assert fresh_optimizer.param_groups[0]["lr"] == 1e-3
