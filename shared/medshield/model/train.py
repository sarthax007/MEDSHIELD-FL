"""Local training loop for the Vision Transformer model.

Task 26 — Build the local training loop.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ModelConfig
from .logging import ExperimentLogger

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the appropriate PyTorch device.

    Returns
    -------
    torch.device
        'cuda' if a GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    loss_fn: nn.Module,
    epochs: int,
    device: Optional[torch.device] = None,
    config: Optional[ModelConfig] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 0,
    experiment_logger: Optional[ExperimentLogger] = None,
) -> nn.Module:
    """Execute the local training loop for the given number of epochs.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    train_loader : DataLoader
        DataLoader for the training split.
    val_loader : DataLoader
        DataLoader for the validation split.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    scheduler : LRScheduler
        The learning rate scheduler.
    loss_fn : nn.Module
        The loss function.
    epochs : int
        Number of epochs to train.
    device : Optional[torch.device], default=None
        Device to run training on. If None, automatically detected.
    config : Optional[ModelConfig], default=None
        The model configuration (required if checkpointing is enabled).
    checkpoint_dir : Optional[str], default=None
        Directory to save checkpoints. If None, checkpointing is skipped.
    checkpoint_interval : int, default=0
        Interval (in epochs) at which to save checkpoints. If 0, only best models are saved.
    experiment_logger : Optional[ExperimentLogger], default=None
        Logger to record metrics per epoch.


    Returns
    -------
    nn.Module
        The trained model.
    """
    if device is None:
        device = get_device()

    model.to(device)

    # If the loss function has a weight tensor, make sure it's on the right device
    if hasattr(loss_fn, "weight") and loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(device)

    logger.info(f"Starting training loop for {epochs} epochs on {device}...")

    use_amp = config is not None and config.mixed_precision and device.type == "cuda"
    if use_amp:
        logger.info("Mixed precision training (AMP) is ENABLED.")
    else:
        logger.info("Mixed precision training (AMP) is DISABLED.")

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -----------------------------------------
        # Training Phase
        # -----------------------------------------
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        avg_train_loss = train_loss / max(1, train_samples)

        # Step the scheduler at the end of the epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # -----------------------------------------
        # Validation Phase
        # -----------------------------------------
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        avg_val_loss = val_loss / max(1, val_samples)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if experiment_logger is not None:
            experiment_logger.log_metrics(
                epoch=epoch,
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                },
            )

        # -----------------------------------------
        # Checkpointing
        # -----------------------------------------
        if checkpoint_dir is not None and config is not None:
            # Model identifier for the checkpoint names
            model_id = config.model_name.lower().replace("/", "-")

            # Save best validation model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = f"{checkpoint_dir}/{model_id}_best_model.pt"
                save_checkpoint(
                    model=model,
                    epoch=epoch,
                    val_loss=avg_val_loss,
                    config=config,
                    filepath=best_ckpt_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            # Save interval checkpoint
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                interval_ckpt_path = (
                    f"{checkpoint_dir}/{model_id}_epoch_{epoch}_val-loss_{avg_val_loss:.4f}.pt"
                )
                save_checkpoint(
                    model=model,
                    epoch=epoch,
                    val_loss=avg_val_loss,
                    config=config,
                    filepath=interval_ckpt_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

    logger.info("Training complete.")
    return model
