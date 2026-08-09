import pytest
import tenseal as ts
from medshield.crypto.context import create_ckks_context


def test_ckks_context_creation():
    """
    Test that the CKKS context is created with the expected parameters for MedShield-FL.
    """
    context = create_ckks_context()
    
    assert context is not None
    assert context.is_private()  # Since secret key is still bound to it
    assert context.has_galois_keys()

    # The TenSEAL context does not currently expose simple Python properties for all 
    # internal BFV/CKKS parameters like coeff_mod_bit_sizes natively in Python bindings, 
    # but we can verify it by checking the global_scale and ensuring encryption works.
    # Note: tenseal < 0.3.14 might not expose `context.global_scale` cleanly, but we set it to 2**40.
    
    scale_value = getattr(context, 'global_scale', None)
    if scale_value is not None:
        # global_scale is stored in context
        assert abs(scale_value - 2**40) < 1e-6, "global_scale should be 2**40"


def test_ckks_context_encryption_decryption_precision():
    """
    Test that encrypting and decrypting a test vector through this context 
    returns values within acceptable CKKS precision.
    """
    context = create_ckks_context()
    
    # Test vector
    plain_vector = [1.12345, 2.71828, -3.14159, 0.00001]
    
    # Encrypt
    encrypted_vector = ts.ckks_vector(context, plain_vector)
    
    # Decrypt
    decrypted_vector = encrypted_vector.decrypt()
    
    # Check precision (tolerance of 1e-4)
    for p, d in zip(plain_vector, decrypted_vector):
        assert abs(p - d) < 1e-4, f"Precision loss too high: expected {p}, got {d}"

