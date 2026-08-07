"""Models package.

Contains the PyTorch ViT models and associated heads.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .delta import apply_delta, apply_model_delta, compute_delta, compute_model_delta
from .logging import ExperimentLogger
from .loss import get_loss_function
from .metrics import compute_metrics
from .optimizer import get_optimizer, get_scheduler
from .registry import create_model
from .serialization import (
    critical_vector_to_model,
    critical_vector_to_state_dict,
    get_critical_ratio,
    get_serialization_layout,
    is_critical_parameter,
    model_to_critical_vector,
    model_to_vector,
    state_dict_to_critical_vector,
    state_dict_to_vector,
    vector_to_model,
    vector_to_state_dict,
)
from .train import get_device, train_model
from .vit import TumorClassifier, create_vit_backbone

__all__ = [
    "ModelConfig",
    "create_model",
    "create_vit_backbone",
    "TumorClassifier",
    "get_loss_function",
    "get_optimizer",
    "get_scheduler",
    "train_model",
    "get_device",
    "compute_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "ExperimentLogger",
    "model_to_vector",
    "vector_to_model",
    "state_dict_to_vector",
    "vector_to_state_dict",
    "get_serialization_layout",
    "is_critical_parameter",
    "model_to_critical_vector",
    "critical_vector_to_model",
    "state_dict_to_critical_vector",
    "critical_vector_to_state_dict",
    "get_critical_ratio",
    "compute_delta",
    "apply_delta",
    "compute_model_delta",
    "apply_model_delta",
]
