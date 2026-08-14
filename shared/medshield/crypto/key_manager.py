from pathlib import Path

import tenseal as ts

from .context import create_ckks_context

PUBLIC_CONTEXT_FILENAME = "public_context.bytes"
SECRET_CONTEXT_FILENAME = "secret_context.bytes"


class SecurityError(Exception):
    """Raised when a security isolation rule is violated."""

    pass


def validate_outbound_context(context: ts.Context) -> None:
    """
    Ensures the given context does not contain a secret key.
    This acts as a safeguard before sending any data to the server.
    """
    if context.has_secret_key():
        raise SecurityError(
            "SECURITY VIOLATION: Attempted to use or serialize a context containing "
            "the secret key for an outbound operation. This compromises private-key isolation."
        )


def generate_and_save_keys(storage_dir: Path | str) -> None:
    """
    Generates a new CKKS context and saves the public context and secret context
    to the specified directory.

    SECURITY RULE:
    The `secret_context.bytes` file must remain at the hospital. It should never
    be shared over the network, as it contains the private key that allows decryption.
    The `public_context.bytes` file can be shared with the central server.
    """
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create the full context (contains the secret key by default)
    context = create_ckks_context()

    # 2. Save the full secret context (contains secret key)
    # We save this so the hospital can simply load it to decrypt.
    secret_context_bytes = context.serialize(
        save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True
    )
    with open(storage_dir / SECRET_CONTEXT_FILENAME, "wb") as f:
        f.write(secret_context_bytes)

    # 3. Drop the secret key from the context to make it public
    context.make_context_public()

    # 4. Save the public context (lacks secret key)
    public_context_bytes = context.serialize(
        save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
    )
    with open(storage_dir / PUBLIC_CONTEXT_FILENAME, "wb") as f:
        f.write(public_context_bytes)


def load_public_context(storage_dir: Path | str) -> ts.Context:
    """
    Loads and returns the public CKKS context.

    This context can be used to ENCRYPT data, perform homomorphic operations,
    but CANNOT decrypt. This is what the central server uses.
    """
    storage_dir = Path(storage_dir)
    with open(storage_dir / PUBLIC_CONTEXT_FILENAME, "rb") as f:
        public_context_bytes = f.read()

    return ts.context_from(public_context_bytes)


def load_secret_context(storage_dir: Path | str) -> ts.Context:
    """
    Loads and returns the full CKKS context including the secret key.

    This returns a fully functional context capable of DECRYPTING.
    This function should ONLY be called within the hospital's secured environment.
    """
    storage_dir = Path(storage_dir)
    with open(storage_dir / SECRET_CONTEXT_FILENAME, "rb") as f:
        secret_context_bytes = f.read()

    return ts.context_from(secret_context_bytes)
