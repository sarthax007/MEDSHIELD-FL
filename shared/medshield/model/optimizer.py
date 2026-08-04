"""Optimizer and Learning Rate Scheduler configuration.

Task 25 — Implement the optimizer and learning-rate scheduler.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from .config import ModelConfig


def get_optimizer(model: nn.Module, config: ModelConfig) -> torch.optim.Optimizer:
    """Build the optimizer for the given model and config.

    Parameters
    ----------
    model : nn.Module
        The model to optimize.
    config : ModelConfig
        The configuration containing hyperparameter settings.

    Returns
    -------
    torch.optim.Optimizer
        An initialized optimizer (AdamW).
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def get_scheduler(optimizer: torch.optim.Optimizer, config: ModelConfig) -> LRScheduler:
    """Build the learning rate scheduler based on config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    config : ModelConfig
        The configuration containing hyperparameter settings (e.g. epochs).

    Returns
    -------
    LRScheduler
        A CosineAnnealingLR scheduler configured for the specified number of epochs.
    """
    # Using CosineAnnealingLR as a standard for ViT fine-tuning
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )
