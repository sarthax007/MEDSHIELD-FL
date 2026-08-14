# Private-Key Isolation Design

This document details the security model and guarantees regarding private-key isolation in MedShield-FL.

## Objective
The core premise of MedShield-FL is that the central server, and any intermediate network actors, must never have the ability to read the raw hospital data or the plaintext model updates. This relies entirely on the isolation of the Homomorphic Encryption (HE) private (secret) key.

## Design Guarantees
1. **Separation of Contexts**:
   When keys are generated locally at a hospital using `generate_and_save_keys`, two contexts are explicitly serialized:
   - `secret_context.bytes`: Contains the fully capable CKKS context (including the private key). This file never leaves the hospital.
   - `public_context.bytes`: Contains the CKKS context but with the secret key explicitly dropped. This is shared with the server and other clients if necessary for operations.

2. **Serialization Safeguards**:
   The code strictly enforces that outbound model updates do not inadvertently leak the private key. Before converting a ciphertext vector to bytes (using `serialize_encrypted_vector`), the attached CKKS context is verified via `validate_outbound_context`. If it detects the presence of a secret key, a `SecurityError` is immediately raised, crashing the process rather than leaking the key.

3. **Server-Side Limitations**:
   The central server only ever loads `public_context.bytes` (if it needs to perform operations like aggregation) and receives ciphertext updates. It is mathematically impossible (within the bounds of CKKS security) for the server to call `.decrypt()` on these ciphertexts without the secret key. Attempts to do so raise errors at the library (TenSEAL) level.

## Threat Model (Server as Honest-but-Curious)
We assume an "honest-but-curious" central server. It correctly follows the orchestration and aggregation protocols, but it might try to inspect the data passing through it.
Because the server possesses only the public key, the Galois keys, and Relin keys, it can add and multiply ciphertexts but cannot decrypt them. Thus, the hospital's local updates remain completely private.

## Testing
This design is backed by automated tests (e.g., `test_key_isolation.py`) that verify:
- Public contexts do not contain a secret key.
- Server-side contexts cannot decrypt ciphertexts.
- The serialization pipeline throws a `SecurityError` if it receives a context with a secret key.
