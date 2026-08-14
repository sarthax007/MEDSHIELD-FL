import tempfile
from pathlib import Path

import torch

from medshield.crypto.aggregation import (
    add_encrypted_vectors,
    add_selective_updates,
    multiply_encrypted_vector_by_scalar,
)
from medshield.crypto.encryption import decrypt_vector, encrypt_vector
from medshield.crypto.key_manager import (
    generate_and_save_keys,
    load_public_context,
    load_secret_context,
)
from medshield.crypto.selective import apply_selective_update, create_selective_update
from medshield.model.config import ModelConfig
from medshield.model.registry import create_model


def test_add_encrypted_vectors():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        # We simulate 3 clients
        vec1 = torch.randn(5000)
        vec2 = torch.randn(5000)
        vec3 = torch.randn(5000)

        # Plaintext sum
        expected_sum = vec1 + vec2 + vec3

        # Encrypt individually using only public context
        enc1 = encrypt_vector(public_ctx, vec1)
        enc2 = encrypt_vector(public_ctx, vec2)
        enc3 = encrypt_vector(public_ctx, vec3)

        # Homomorphic addition (requires NO secret key)
        summed_enc = add_encrypted_vectors([enc1, enc2, enc3])

        # Decrypt to verify
        for chunk in summed_enc:
            chunk.link_context(secret_ctx)

        decrypted_sum_list = decrypt_vector(summed_enc)
        decrypted_sum = torch.tensor(decrypted_sum_list)

        # Verify tolerance
        max_diff = torch.max(torch.abs(expected_sum - decrypted_sum)).item()
        assert max_diff < 1e-3, f"Homomorphic sum diverged from plaintext sum: {max_diff}"


def test_add_selective_updates():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        # Simulate 3 models (clients)
        config = ModelConfig(num_classes=2, pretrained=False)
        m1 = create_model(config)
        m2 = create_model(config)
        m3 = create_model(config)

        u1 = create_selective_update(m1, public_ctx)
        u2 = create_selective_update(m2, public_ctx)
        u3 = create_selective_update(m3, public_ctx)

        # Homomorphically add selective updates
        summed_update = add_selective_updates([u1, u2, u3])

        # Create a model to apply the update to
        m_sum = create_model(config)
        apply_selective_update(summed_update, m_sum, secret_ctx)

        # Verify non-critical part manually
        nc_sum = u1.plaintext_non_critical + u2.plaintext_non_critical + u3.plaintext_non_critical
        assert torch.allclose(summed_update.plaintext_non_critical, nc_sum, atol=1e-5)

        # Critical part is tested by test_add_encrypted_vectors and successful apply_selective_update


def test_multiply_encrypted_vector():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        vec = torch.randn(5000)
        scalar = 2.5
        expected_scaled = vec * scalar

        enc_vec = encrypt_vector(public_ctx, vec)
        scaled_enc_vec = multiply_encrypted_vector_by_scalar(enc_vec, scalar)

        for chunk in scaled_enc_vec:
            chunk.link_context(secret_ctx)

        decrypted_scaled_list = decrypt_vector(scaled_enc_vec)
        decrypted_scaled = torch.tensor(decrypted_scaled_list)

        max_diff = torch.max(torch.abs(expected_scaled - decrypted_scaled)).item()
        assert max_diff < 1e-3, f"Homomorphic multiplication diverged from plaintext: {max_diff}"


def test_encrypted_weighted_average():
    from medshield.crypto.aggregation import (
        add_encrypted_vectors,
        multiply_encrypted_vector_by_scalar,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        # Simulating 3 clients with different weights
        vec1 = torch.randn(5000)
        w1 = 0.5
        vec2 = torch.randn(5000)
        w2 = 0.3
        vec3 = torch.randn(5000)
        w3 = 0.2

        # Plaintext weighted average
        expected_wavg = (vec1 * w1) + (vec2 * w2) + (vec3 * w3)

        # Encrypted weighted average
        enc1 = encrypt_vector(public_ctx, vec1)
        enc2 = encrypt_vector(public_ctx, vec2)
        enc3 = encrypt_vector(public_ctx, vec3)

        scaled_enc1 = multiply_encrypted_vector_by_scalar(enc1, w1)
        scaled_enc2 = multiply_encrypted_vector_by_scalar(enc2, w2)
        scaled_enc3 = multiply_encrypted_vector_by_scalar(enc3, w3)

        # Add them up
        wavg_enc = add_encrypted_vectors([scaled_enc1, scaled_enc2, scaled_enc3])

        for chunk in wavg_enc:
            chunk.link_context(secret_ctx)

        decrypted_wavg_list = decrypt_vector(wavg_enc)
        decrypted_wavg = torch.tensor(decrypted_wavg_list)

        max_diff = torch.max(torch.abs(expected_wavg - decrypted_wavg)).item()
        assert max_diff < 1e-3, f"Homomorphic weighted average diverged: {max_diff}"
