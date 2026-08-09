import tempfile
from pathlib import Path

import torch

from medshield.crypto.encryption import decrypt_vector, encrypt_vector
from medshield.crypto.key_manager import (
    generate_and_save_keys,
    load_public_context,
    load_secret_context,
)


def test_encrypt_vector_with_chunking_and_semantic_security():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Generate keys
        generate_and_save_keys(temp_path)

        # 2. Load contexts
        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        # Create a tensor larger than 4096 to ensure chunking works
        # e.g., 5000 elements
        test_tensor = torch.randn(5000)

        # 3. Encrypt using public context (no secret key!)
        encrypted_chunks_1 = encrypt_vector(public_ctx, test_tensor)

        # Verify it created 2 chunks (since 5000 > 4096)
        assert len(encrypted_chunks_1) == 2

        # 4. Encrypt the same vector again to verify semantic security
        encrypted_chunks_2 = encrypt_vector(public_ctx, test_tensor)

        # Extract the actual bytes to verify they are distinct
        c1_bytes = encrypted_chunks_1[0].serialize()
        c2_bytes = encrypted_chunks_2[0].serialize()
        assert (
            c1_bytes != c2_bytes
        ), "Semantic security failed: identical ciphertexts for identical plaintexts"

        # 5. Decrypt both and ensure they match the original
        # Note: We must link the secret context before decrypting
        for c in encrypted_chunks_1:
            c.link_context(secret_ctx)
        for c in encrypted_chunks_2:
            c.link_context(secret_ctx)

        decrypted_1 = decrypt_vector(encrypted_chunks_1)
        decrypted_2 = decrypt_vector(encrypted_chunks_2)

        assert len(decrypted_1) == 5000
        assert len(decrypted_2) == 5000

        # Verify precision
        original_list = test_tensor.tolist()
        for orig, d1, d2 in zip(original_list, decrypted_1, decrypted_2):
            assert abs(orig - d1) < 1e-3
            assert abs(orig - d2) < 1e-3
