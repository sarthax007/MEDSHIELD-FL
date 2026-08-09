import pytest
import tempfile
import torch
from pathlib import Path

from medshield.crypto.key_manager import (
    generate_and_save_keys,
    load_public_context,
    load_secret_context
)
from medshield.crypto.encryption import encrypt_vector, decrypt_vector
from medshield.model.registry import create_model
from medshield.model.config import ModelConfig
from medshield.model.serialization import (
    model_to_critical_vector,
    critical_vector_to_model
)

def test_decryption_requires_secret_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)
        
        public_ctx = load_public_context(temp_path)
        
        test_tensor = torch.randn(100)
        encrypted_chunks = encrypt_vector(public_ctx, test_tensor)
        
        # public_ctx lacks the secret key, so this should fail
        with pytest.raises(ValueError, match="doesn't hold a secret_key"):
            decrypt_vector(encrypted_chunks)

def test_decryption_reconstruction_error():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)
        
        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)
        
        # Test vector of size > 4096 to ensure we test multiple chunks
        test_tensor = torch.randn(5000)
        
        # Encrypt
        encrypted_chunks = encrypt_vector(public_ctx, test_tensor)
        
        # Bind secret key and decrypt
        for chunk in encrypted_chunks:
            chunk.link_context(secret_ctx)
            
        decrypted_list = decrypt_vector(encrypted_chunks)
        decrypted_tensor = torch.tensor(decrypted_list)
        
        # Calculate errors
        absolute_errors = torch.abs(test_tensor - decrypted_tensor)
        mean_error = torch.mean(absolute_errors).item()
        max_error = torch.max(absolute_errors).item()
        
        # Our context has 40 bits of scale, so tolerance should be very good (e.g. 1e-4)
        assert mean_error < 1e-4, f"Mean error {mean_error} is too high"
        assert max_error < 1e-3, f"Max error {max_error} is too high"

def test_decryption_preserves_model_predictions():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)
        
        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)
        
        # Create a small instance of the model for testing
        config = ModelConfig(num_classes=2, pretrained=False)
        model = create_model(config)
        model.eval()
        
        # Create dummy input batch
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = model(dummy_input)
            
        # Extract critical parameters
        critical_vec = model_to_critical_vector(model)
        
        # Encrypt
        encrypted_chunks = encrypt_vector(public_ctx, critical_vec)
        
        # Decrypt
        for chunk in encrypted_chunks:
            chunk.link_context(secret_ctx)
        decrypted_list = decrypt_vector(encrypted_chunks)
        decrypted_tensor = torch.tensor(decrypted_list)
        
        # Apply back to a fresh model (or same model)
        critical_vector_to_model(decrypted_tensor, model)
        
        # Forward pass again
        with torch.no_grad():
            new_output = model(dummy_input)
            
        # The logits should be almost identical
        max_logit_diff = torch.max(torch.abs(original_output - new_output)).item()
        assert max_logit_diff < 1e-3, f"Logits changed too much: diff {max_logit_diff}"
