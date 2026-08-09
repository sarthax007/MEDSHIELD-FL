import tempfile
from pathlib import Path

import pytest
import tenseal as ts

from medshield.crypto.key_manager import (
    PUBLIC_CONTEXT_FILENAME,
    SECRET_CONTEXT_FILENAME,
    generate_and_save_keys,
    load_public_context,
    load_secret_context,
)


def test_key_manager_generation_and_loading():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Generate keys
        generate_and_save_keys(temp_path)

        # Verify files exist
        assert (temp_path / PUBLIC_CONTEXT_FILENAME).exists()
        assert (temp_path / SECRET_CONTEXT_FILENAME).exists()

        # 2. Load public context and verify it cannot decrypt
        public_ctx = load_public_context(temp_path)
        assert not public_ctx.is_private()

        # We can encrypt with the public context
        encrypted_vec = ts.ckks_vector(public_ctx, [1.0, 2.0])

        # But we cannot decrypt with it
        with pytest.raises(ValueError, match="doesn't hold a secret_key"):
            encrypted_vec.decrypt()

        # 3. Load secret context and verify it can decrypt
        secret_ctx = load_secret_context(temp_path)
        assert secret_ctx.is_private()

        # Link the previously encrypted vector to the secret context so we can decrypt it
        encrypted_vec.link_context(secret_ctx)
        decrypted_vec = encrypted_vec.decrypt()

        assert abs(decrypted_vec[0] - 1.0) < 1e-4
        assert abs(decrypted_vec[1] - 2.0) < 1e-4
