"""CKKS context, key management, encryption and homomorphic aggregation (Tasks 35-46)."""

from .context import create_ckks_context
from .encryption import decrypt_vector, encrypt_vector
from .key_manager import generate_and_save_keys, load_public_context, load_secret_context

__all__ = [
    "create_ckks_context",
    "generate_and_save_keys",
    "load_public_context",
    "load_secret_context",
    "encrypt_vector",
    "decrypt_vector",
]
