# Threat Model: Server-Side Attacker

## Scope
This document outlines the threat model concerning the central server in the MedShield-FL architecture. It specifically addresses what a malicious actor or compromised server can and cannot see during the federated learning process.

## Attacker Capabilities
An attacker with full access to the central server can:
- Intercept all incoming payloads from hospital clients.
- View the aggregated ciphertexts before they are sent back.
- Inspect the public keys, Galois keys, and Relin keys provided by the clients.

## What the Server CANNOT See
Due to the use of Homomorphic Encryption (CKKS via TenSEAL):
- **Raw Patient Data:** The server never receives raw patient images or labels. Data remains strictly on the client side.
- **Plaintext Model Updates:** The server only receives encrypted model updates (ciphertexts). It does not possess the secret key needed to decrypt them.
- **Client-Specific Signals:** The server aggregates ciphertexts homomorphically. Even if it attempts to decrypt a single client's update or the aggregated model, the operation will fail or yield garbage because the server lacks the secret key.

## Assertions
As proven by automated security tests (see `test_server_security.py`), any attempt to decrypt ciphertexts using only the server's available contexts (public keys) results in a cryptographic failure (`ValueError` indicating a missing secret key).

Therefore, confidentiality of both patient data and local model updates is preserved against a curious or malicious server.
