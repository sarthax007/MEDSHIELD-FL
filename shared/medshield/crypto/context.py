import tenseal as ts


def create_ckks_context() -> ts.Context:
    """
    Builds a reusable TenSEAL context configured for CKKS homomorphic encryption.

    Parameters used for MedShield-FL:
    - poly_modulus_degree: 8192
        Provides a 128-bit security level against known attacks while keeping
        operations relatively fast and ciphertext sizes manageable.
    - coeff_mod_bit_sizes: [60, 40, 40, 60]
        Determines the multiplicative depth of the encryption context.
        The middle moduli (40, 40) provide the capacity for consecutive
        multiplications. For FedAvg, we only need a depth of 1 (to perform
        weighted scalar multiplication by the hospital's sample count).
    - global_scale: 2**40
        Provides 40 bits of precision for the fractional part of numbers. This
        ensures that the small floating-point updates (deltas) of the model
        are represented accurately during homomorphic operations.

    Returns:
        A fully initialized `tenseal.Context` object with Galois keys generated,
        which are required for operations like vector rotations (and generally
        good practice to have for tensor operations).
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40

    # Generate Galois keys (needed for some vector operations)
    context.generate_galois_keys()

    return context
