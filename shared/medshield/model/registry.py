"""Model registry and factory functions.

Task 23 — Implement a model configuration and registry.
"""

import torch.nn as nn

from .config import ModelConfig
from .vit import TumorClassifier


def create_model(config: ModelConfig) -> nn.Module:
    """Build a model from a ModelConfig instance.

    Parameters
    ----------
    config : ModelConfig
        The configuration object specifying hyperparameters.

    Returns
    -------
    nn.Module
        The instantiated PyTorch model ready for training or inference.
    """
    if config.model_name == "vit_base_patch16_224":
        return TumorClassifier(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            drop_rate=config.drop_rate,
        )
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")
