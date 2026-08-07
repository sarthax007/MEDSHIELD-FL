"""Model weights serialization and deserialization utilities.

Provides functions to convert a model's state_dict to and from a flat 1D
vector. It guarantees deterministic ordering by sorting the parameter keys,
which ensures consistent indexing across all federated clients.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def state_dict_to_vector(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a model's state_dict into a flat 1-D tensor.

    Keys are sorted alphabetically to guarantee a deterministic ordering
    of parameters.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The state dictionary of the model to flatten.

    Returns
    -------
    torch.Tensor
        A flat 1-D CPU tensor containing all the weights.
    """
    sorted_keys = sorted(state_dict.keys())
    tensors = []
    for key in sorted_keys:
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            tensors.append(val.detach().cpu().flatten())
        else:
            logger.warning(f"Key {key} in state_dict is not a torch.Tensor. Skipping.")

    if not tensors:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(tensors)


def vector_to_state_dict(
    vector: torch.Tensor, template_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Reconstruct a state_dict from a flat 1-D tensor using a template.

    Parameters
    ----------
    vector : torch.Tensor
        The flat 1-D tensor containing the model weights.
    template_state_dict : dict[str, torch.Tensor]
        A template state_dict defining keys, shapes, devices, and data types.

    Returns
    -------
    dict[str, torch.Tensor]
        A reconstructed state dictionary matching the structure of the template.

    Raises
    ------
    ValueError
        If the vector length does not match the total number of parameter elements.
    """
    sorted_keys = sorted(template_state_dict.keys())
    reconstructed = {}
    current_idx = 0

    for key in sorted_keys:
        val = template_state_dict[key]
        if isinstance(val, torch.Tensor):
            numel = val.numel()
            flat_slice = vector[current_idx : current_idx + numel]
            if len(flat_slice) != numel:
                raise ValueError(
                    f"Vector is too short. Expected at least {current_idx + numel} elements "
                    f"but vector ended at {current_idx + len(flat_slice)} while restoring key '{key}'."
                )
            # Reconstruct original shape, device, and dtype
            reconstructed[key] = flat_slice.view(val.shape).to(val.device).type(val.dtype)
            current_idx += numel
        else:
            reconstructed[key] = val

    if current_idx < len(vector):
        raise ValueError(
            f"Vector has extra elements. Used {current_idx} out of {len(vector)} elements."
        )

    return reconstructed


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten a PyTorch model's state_dict parameters/buffers into a 1-D tensor.

    Parameters
    ----------
    model : nn.Module
        The model whose weights are to be serialized.

    Returns
    -------
    torch.Tensor
        A flat 1-D CPU tensor of weights.
    """
    return state_dict_to_vector(model.state_dict())


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    """Load a flat 1-D tensor of weights back into a PyTorch model in-place.

    Parameters
    ----------
    vector : torch.Tensor
        The flat 1-D tensor containing the weights to load.
    model : nn.Module
        The model to load the weights into.
    """
    new_state_dict = vector_to_state_dict(vector, model.state_dict())
    model.load_state_dict(new_state_dict)


def get_serialization_layout(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Document the layout of the flat vector representation.

    Provides mapping metadata including parameter name, shape, size (numel),
    and offset start/end index within the serialized 1-D vector.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The state dict to analyze.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries containing layout info:
        - 'name': str, key of the parameter
        - 'shape': list[int], original shape of the tensor
        - 'numel': int, number of elements
        - 'start_offset': int, starting index in the flat vector
        - 'end_offset': int, ending index (exclusive) in the flat vector
    """
    sorted_keys = sorted(state_dict.keys())
    layout = []
    current_offset = 0

    for key in sorted_keys:
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            numel = val.numel()
            layout.append(
                {
                    "name": key,
                    "shape": list(val.shape),
                    "numel": numel,
                    "start_offset": current_offset,
                    "end_offset": current_offset + numel,
                }
            )
            current_offset += numel

    return layout
