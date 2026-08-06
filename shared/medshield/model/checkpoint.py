"""Model checkpointing utilities.

Task 28 \u2014 Implement checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_loss: float,
    config: ModelConfig,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """Save a model checkpoint.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to save.
    epoch : int
        The current training epoch.
    val_loss : float
        The validation loss at this epoch.
    config : ModelConfig
        The model configuration used to construct the model.
    filepath : str
        The path where the checkpoint will be saved.
    optimizer : Optional[torch.optim.Optimizer], default=None
        The optimizer state to save.
    scheduler : Optional[Any], default=None
        The learning rate scheduler state to save.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config_dict": config.to_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[ModelConfig, int, float]:
    """Load a model checkpoint.

    Parameters
    ----------
    filepath : str
        The path to the checkpoint file.
    model : nn.Module
        The PyTorch model into which the weights will be loaded.
    optimizer : Optional[torch.optim.Optimizer], default=None
        The optimizer to resume (if provided and saved in checkpoint).
    scheduler : Optional[Any], default=None
        The scheduler to resume (if provided and saved in checkpoint).

    Returns
    -------
    Tuple[ModelConfig, int, float]
        The model configuration, epoch, and validation loss from the checkpoint.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    config = ModelConfig.from_dict(checkpoint["config_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]

    logger.info(f"Loaded checkpoint from {filepath} (Epoch: {epoch}, Val Loss: {val_loss:.4f})")
    return config, epoch, val_loss
