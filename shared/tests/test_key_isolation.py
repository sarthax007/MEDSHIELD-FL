import tempfile
from pathlib import Path

import pytest

from medshield.crypto.encryption import encrypt_vector
from medshield.crypto.key_manager import (
    SecurityError,
    generate_and_save_keys,
    load_public_context,
    load_secret_context,
    validate_outbound_context,
)
from medshield.crypto.serialization import serialize_encrypted_vector


def test_public_context_lacks_secret_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        assert not public_ctx.has_secret_key(), "Public context should NOT have a secret key!"

        # validate_outbound_context should succeed
        validate_outbound_context(public_ctx)


def test_server_cannot_decrypt():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)

        # Client encrypts data with public context
        plaintext = [1.0, 2.0, 3.0]
        encrypted = encrypt_vector(public_ctx, plaintext)

        # Attempt to decrypt the vector with the public context (simulating the server)
        # We expect an error from TenSEAL indicating missing secret key
        with pytest.raises(ValueError, match="doesn't hold a secret_key"):
            encrypted[0].decrypt()


def test_outbound_serialization_fails_with_secret_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        secret_ctx = load_secret_context(temp_path)

        # Test validate_outbound_context directly
        with pytest.raises(SecurityError, match="SECURITY VIOLATION"):
            validate_outbound_context(secret_ctx)

        plaintext = [1.0, 2.0, 3.0]
        # We encrypt with secret_ctx just for testing serialization (which might happen)
        # The vector will be linked to secret_ctx
        encrypted = encrypt_vector(secret_ctx, plaintext)

        with pytest.raises(SecurityError, match="SECURITY VIOLATION"):
            serialize_encrypted_vector(encrypted)
