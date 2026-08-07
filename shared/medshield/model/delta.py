"""Model update delta utilities.

Task 34 — Extract model update deltas.

Provides functions to compute the delta (difference) between a global model
and a locally trained model, and to apply that delta back to the global model.
This delta represents the client's contribution in a federated learning round.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medshield.model.serialization import (
    critical_vector_to_model,
    model_to_critical_vector,
    model_to_vector,
    vector_to_model,
)


def compute_delta(global_vector: torch.Tensor, local_vector: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise difference (delta) between local and global vectors.

    The delta is defined as: delta = local - global.

    Parameters
    ----------
    global_vector : torch.Tensor
        The flattened weights of the global model.
    local_vector : torch.Tensor
        The flattened weights of the locally trained model.

    Returns
    -------
    torch.Tensor
        The computed delta vector.
    """
    if global_vector.shape != local_vector.shape:
        raise ValueError(
            f"Shape mismatch: global {global_vector.shape} != local {local_vector.shape}"
        )
    return local_vector - global_vector


def apply_delta(global_vector: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply a delta update to a global vector.

    The new vector is defined as: new_global = global + delta.

    Parameters
    ----------
    global_vector : torch.Tensor
        The flattened weights of the global model.
    delta : torch.Tensor
        The delta update to apply.

    Returns
    -------
    torch.Tensor
        The updated global vector.
    """
    if global_vector.shape != delta.shape:
        raise ValueError(
            f"Shape mismatch: global {global_vector.shape} != delta {delta.shape}"
        )
    return global_vector + delta


def compute_model_delta(
    global_model: nn.Module, local_model: nn.Module, critical_only: bool = False
) -> torch.Tensor:
    """Compute the delta update between a global and local model.

    Parameters
    ----------
    global_model : nn.Module
        The global model before local training.
    local_model : nn.Module
        The model after local training.
    critical_only : bool
        If True, computes the delta only for critical parameters (Task 33).
        Otherwise, computes it for the entire model.

    Returns
    -------
    torch.Tensor
        A flat 1-D CPU tensor representing the update delta.
    """
    if critical_only:
        global_vec = model_to_critical_vector(global_model)
        local_vec = model_to_critical_vector(local_model)
    else:
        global_vec = model_to_vector(global_model)
        local_vec = model_to_vector(local_model)

    return compute_delta(global_vec, local_vec)


def apply_model_delta(
    global_model: nn.Module, delta: torch.Tensor, critical_only: bool = False
) -> None:
    """Apply a delta update directly to a global model in-place.

    Parameters
    ----------
    global_model : nn.Module
        The global model to update.
    delta : torch.Tensor
        The delta update vector to apply.
    critical_only : bool
        If True, applies the delta only to critical parameters (Task 33).
        Otherwise, applies it to the entire model.
    """
    if critical_only:
        global_vec = model_to_critical_vector(global_model)
        new_vec = apply_delta(global_vec, delta)
        critical_vector_to_model(new_vec, global_model)
    else:
        global_vec = model_to_vector(global_model)
        new_vec = apply_delta(global_vec, delta)
        vector_to_model(new_vec, global_model)
