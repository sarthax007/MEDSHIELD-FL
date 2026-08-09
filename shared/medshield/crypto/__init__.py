"""CKKS context, key management, encryption and homomorphic aggregation (Tasks 35-46)."""

from .aggregation import (
    add_encrypted_vectors,
    add_selective_updates,
    multiply_encrypted_vector_by_scalar,
    multiply_selective_update_by_scalar,
)
from .context import create_ckks_context
from .encryption import decrypt_vector, encrypt_vector
from .key_manager import generate_and_save_keys, load_public_context, load_secret_context
from .selective import SelectiveUpdate, apply_selective_update, create_selective_update

__all__ = [
    "create_ckks_context",
    "generate_and_save_keys",
    "load_public_context",
    "load_secret_context",
    "encrypt_vector",
    "decrypt_vector",
    "SelectiveUpdate",
    "create_selective_update",
    "apply_selective_update",
    "add_encrypted_vectors",
    "add_selective_updates",
    "multiply_encrypted_vector_by_scalar",
    "multiply_selective_update_by_scalar",
]
