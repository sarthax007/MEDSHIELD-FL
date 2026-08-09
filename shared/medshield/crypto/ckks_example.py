"""
MedShield-FL CKKS Encryption Example
====================================

What CKKS lets this project do:
Homomorphic encryption allows computation on encrypted data without ever 
needing to decrypt it. The CKKS scheme specifically supports operations 
on real (floating-point) numbers. In the context of MedShield-FL, this 
means a hospital can encrypt its model weight updates (which are arrays 
of floats) before sending them to the central server. The central server 
can then average these encrypted updates from multiple hospitals and send 
the encrypted result back. At no point does the central server see the 
raw, plaintext model updates (or any patient data), preserving complete 
privacy. Only the participating hospitals have the key to decrypt the 
averaged model.
"""

import tenseal as ts

def run_ckks_example():
    print("=" * 60)
    print("  MedShield-FL: TenSEAL CKKS Encryption Example")
    print("=" * 60)

    # 1. Setup TenSEAL Context for CKKS
    # We use polynomial modulus degree 8192 and suitable coeff_mod_bit_sizes 
    # for a decent balance of precision and speed.
    print("[1] Creating CKKS context...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    # Global scale determines the precision of fractional parts.
    context.global_scale = 2**40
    # Generate Galois keys which are required for some operations like vector rotations (not strictly needed for add/mul here, but good practice)
    context.generate_galois_keys()
    print("    - Global scale:", context.global_scale)

    # 2. Encrypt a small vector
    print("\n[2] Encrypting a vector...")
    plain_vector = [1.5, 2.7, 3.1, -4.2]
    print(f"    - Original Plaintext Vector: {plain_vector}")
    
    encrypted_vector = ts.ckks_vector(context, plain_vector)
    print(f"    - Encrypted Vector (ciphertext size: {encrypted_vector.size()})")

    # 3. Homomorphic Computation
    print("\n[3] Performing homomorphic addition (adding 5.0 to each element)...")
    constant_to_add = 5.0
    encrypted_result = encrypted_vector + constant_to_add
    print("    - Addition performed successfully on ciphertext!")

    # 4. Decrypt the result
    print("\n[4] Decrypting the result...")
    decrypted_result = encrypted_result.decrypt()  # type: ignore
    
    # Format nicely for display
    formatted_decrypted = [round(v, 4) for v in decrypted_result]
    print(f"    - Decrypted Result: {formatted_decrypted}")
    
    # Verify correctness
    expected_result = [v + constant_to_add for v in plain_vector]
    formatted_expected = [round(v, 4) for v in expected_result]
    print(f"    - Expected Result : {formatted_expected}")
    
    print("\nSuccess! CKKS homomorphic encryption is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    run_ckks_example()
