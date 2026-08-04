"""Vision Transformer Backbone.

Task 21 — Load the ViT-Base/16 backbone.

This module provides the ViT backbone architecture using `timm`.
It strips the classification head to output pooled features instead.
"""

import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_vit_backbone(pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Create a ViT-Base/16 backbone.

    Parameters
    ----------
    pretrained : bool
        If True, loads ImageNet pretrained weights.
        If False, initializes randomly.
    drop_rate : float
        Dropout rate.

    Returns
    -------
    nn.Module
        The ViT model without a classification head.
        A forward pass yields a tensor of shape (batch_size, 768).
    """
    # Using timm, num_classes=0 strips the classifier head
    # The output will be the pooled features.
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=pretrained, num_classes=0, drop_rate=drop_rate
    )

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded ViT-Base/16 backbone (Pretrained={pretrained}, DropRate={drop_rate})")
    logger.info(f"Total Parameters: {total_params:,}")

    return model


class TumorClassifier(nn.Module):
    """Vision Transformer with a custom classification head.

    Task 22 — Add the tumor classification head.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()
        self.num_classes = num_classes

        # Instantiate the backbone (outputs shape: [batch_size, 768])
        self.backbone = create_vit_backbone(pretrained=pretrained, drop_rate=drop_rate)

        # Create a new, randomly initialized classification head
        self.head = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, channels, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
