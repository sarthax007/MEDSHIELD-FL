"""Model weights serialization and deserialization utilities.

Task 32 — Serialize / deserialize model weights as a flat vector.
Task 33 — Identify critical parameters for selective encryption.

Provides functions to convert a model's state_dict to and from a flat 1D
vector. It guarantees deterministic ordering by sorting the parameter keys,
which ensures consistent indexing across all federated clients.

Also provides selective extraction of **critical parameters** — those most
relevant to the classification decision (``cls_token`` and ``head``) — for
targeted homomorphic encryption.  Only ~0.003 % of the total ViT-Base/16
parameters are critical, making full HE on just these weights feasible.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critical-parameter identification
# ---------------------------------------------------------------------------

#: Substrings used to identify classification-critical state_dict keys.
#: ``cls_token``  — the learnable [CLS] embedding that aggregates image info.
#: ``head``       — the final linear classifier (weight + bias).
CRITICAL_KEY_PATTERNS: tuple[str, ...] = ("cls_token", "head")


def is_critical_parameter(name: str) -> bool:
    """Return True if *name* belongs to a classification-critical parameter.

    A parameter is deemed critical if its key contains any substring listed
    in :data:`CRITICAL_KEY_PATTERNS` (currently ``cls_token`` and ``head``).

    Parameters
    ----------
    name : str
        The state_dict key to check.

    Returns
    -------
    bool
        ``True`` if the key matches a critical pattern.
    """
    return any(pattern in name for pattern in CRITICAL_KEY_PATTERNS)


# ---------------------------------------------------------------------------
# Full-model serialization (Task 32)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Selective (critical-only) serialization  (Task 33)
# ---------------------------------------------------------------------------


def state_dict_to_critical_vector(
    state_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Extract only the critical parameters and flatten them into a 1-D tensor.

    Critical parameters are identified by :func:`is_critical_parameter`.
    Keys are sorted alphabetically for deterministic ordering.

    For the default ViT-Base/16 + TumorClassifier architecture the critical
    keys are (in sorted order):

    ============================================  =============  ======
    Key                                           Shape          Numel
    ============================================  =============  ======
    ``backbone.cls_token``                        (1, 1, 768)      768
    ``head.bias``                                 (2,)               2
    ``head.weight``                               (2, 768)       1 536
    ============================================  =============  ======

    Total critical elements: **2 306** (≈ 0.003 % of 85 800 194).

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The full state dictionary of the model.

    Returns
    -------
    torch.Tensor
        A flat 1-D CPU tensor containing only the critical weights.
    """
    critical_sd = {k: v for k, v in state_dict.items() if is_critical_parameter(k)}
    return state_dict_to_vector(critical_sd)


def critical_vector_to_state_dict(
    vector: torch.Tensor,
    template_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge a critical-parameter vector back into a full state_dict.

    Non-critical parameters are copied unchanged from *template_state_dict*.
    Critical parameters are restored from *vector* in sorted-key order.

    Parameters
    ----------
    vector : torch.Tensor
        Flat 1-D tensor of critical weights (same order produced by
        :func:`state_dict_to_critical_vector`).
    template_state_dict : dict[str, torch.Tensor]
        The full state_dict supplying shapes for critical keys and values
        for all non-critical keys.

    Returns
    -------
    dict[str, torch.Tensor]
        A complete state_dict with critical parameters replaced from
        *vector* and all other parameters unchanged.

    Raises
    ------
    ValueError
        If *vector* length does not match the total critical-parameter count.
    """
    critical_template = {k: v for k, v in template_state_dict.items() if is_critical_parameter(k)}
    restored_critical = vector_to_state_dict(vector, critical_template)

    merged = {}
    for key in template_state_dict:
        if is_critical_parameter(key):
            merged[key] = restored_critical[key]
        else:
            merged[key] = template_state_dict[key]

    return merged


def model_to_critical_vector(model: nn.Module) -> torch.Tensor:
    """Extract critical parameters from a model as a flat 1-D tensor.

    Parameters
    ----------
    model : nn.Module
        The model whose critical weights are to be extracted.

    Returns
    -------
    torch.Tensor
        A flat 1-D CPU tensor of critical weights.
    """
    return state_dict_to_critical_vector(model.state_dict())


def critical_vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    """Load a critical-parameter vector back into a model in-place.

    Non-critical parameters remain unchanged.

    Parameters
    ----------
    vector : torch.Tensor
        The flat 1-D tensor of critical weights.
    model : nn.Module
        The model to update.
    """
    new_sd = critical_vector_to_state_dict(vector, model.state_dict())
    model.load_state_dict(new_sd)


def get_critical_ratio(model: nn.Module) -> float:
    """Calculate the ratio of critical parameters to total model parameters.

    Logs a human-readable summary to the module logger at INFO level.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    float
        Fraction of parameters that are critical (0.0 – 1.0).
    """
    sd = model.state_dict()
    total_numel = 0
    critical_numel = 0

    for key, val in sd.items():
        if isinstance(val, torch.Tensor):
            numel = val.numel()
            total_numel += numel
            if is_critical_parameter(key):
                critical_numel += numel

    ratio = critical_numel / total_numel if total_numel > 0 else 0.0

    logger.info(
        "Critical-parameter ratio: %d / %d = %.6f%% (%d critical elements)",
        critical_numel,
        total_numel,
        ratio * 100,
        critical_numel,
    )

    return ratio
