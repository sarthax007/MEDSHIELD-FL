import tempfile
from pathlib import Path

import pytest

from medshield.crypto.encryption import encrypt_vector
from medshield.crypto.key_manager import (
    generate_and_save_keys,
    load_public_context,
)


def test_server_cannot_decrypt_client_ciphertext():
    """
    Test confirms the server, with only its available keys, cannot decrypt any client ciphertext.
    Attempting decryption server-side raises an error or yields garbage.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        # Server only has access to the public context
        public_ctx = load_public_context(temp_path)

        # Client encrypts their model update
        plaintext_update = [0.1, -0.5, 0.88, 1.2]
        encrypted_update = encrypt_vector(public_ctx, plaintext_update)

        # Server attempts to decrypt
        for ciphertext in encrypted_update:
            with pytest.raises(ValueError, match="doesn't hold a secret_key"):
                # Decryption should fail because the server context lacks the secret key
                ciphertext.decrypt()


def test_server_context_has_no_secret_key():
    """
    Ensure the server context explicitly does not contain the secret key.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        assert not public_ctx.has_secret_key(), "Server context MUST NOT contain a secret key"
