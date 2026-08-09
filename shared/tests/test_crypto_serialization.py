import pytest
import tenseal as ts
import tempfile
from pathlib import Path
import logging

from medshield.crypto.key_manager import (
    generate_and_save_keys, 
    load_public_context, 
    load_secret_context
)
from medshield.crypto.encryption import encrypt_vector, decrypt_vector
from medshield.crypto.serialization import serialize_encrypted_vector, deserialize_encrypted_vector

def test_ciphertext_serialization(caplog):
    """Test serialization and deserialization of ciphertexts (Task 43)."""
    caplog.set_level(logging.INFO)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)
        
        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)
        
        original_vector = [1.5, -2.3, 3.14, 0.0]
        # Encrypt using public key only
        encrypted = encrypt_vector(public_ctx, original_vector)
        
        # Serialize to bytes
        serialized = serialize_encrypted_vector(encrypted)
        
        # Check that it serialized correctly
        assert isinstance(serialized, list)
        assert len(serialized) == len(encrypted)
        assert isinstance(serialized[0], bytes)
        
        # The serialised size is measured and logged (Task 43 AC 3)
        assert "Serialized encrypted vector size:" in caplog.text
        
        # Deserialize from bytes using public context (Task 43 AC 1)
        deserialized = deserialize_encrypted_vector(public_context=public_ctx, serialized_vector=serialized)
        
        # Serialisation excludes the secret key so transferred bytes cannot be decrypted by a third party (Task 43 AC 4)
        # Verify that without secret key, it cannot be decrypted
        with pytest.raises(ValueError, match="doesn't hold a secret_key"):
            decrypt_vector(deserialized)
            
        # Now link the secret context so we can decrypt
        for chunk in deserialized:
            chunk.link_context(secret_ctx)
            
        # A deserialised ciphertext still decrypts to the correct result (Task 43 AC 2)
        decrypted = decrypt_vector(deserialized)
        
        for orig, dec in zip(original_vector, decrypted):
            assert abs(orig - dec) < 1e-3
