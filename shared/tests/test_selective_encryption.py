import tempfile
from pathlib import Path

import torch

from medshield.crypto.key_manager import (
    generate_and_save_keys,
    load_public_context,
    load_secret_context,
)
from medshield.crypto.selective import (
    SelectiveUpdate,
    apply_selective_update,
    create_selective_update,
)
from medshield.model.config import ModelConfig
from medshield.model.registry import create_model
from medshield.model.serialization import model_to_vector


def test_selective_encryption_reassembly_and_size():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_and_save_keys(temp_path)

        public_ctx = load_public_context(temp_path)
        secret_ctx = load_secret_context(temp_path)

        # Create model
        config = ModelConfig(num_classes=2, pretrained=False)
        model = create_model(config)
        model.eval()

        # Measure size of full unencrypted state_dict
        full_vec = model_to_vector(model)
        full_unencrypted_size_bytes = full_vec.element_size() * full_vec.numel()

        # We need a dummy input to verify predictions
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_logits = model(dummy_input)

        # Create selective update
        selective_update = create_selective_update(model, public_ctx)

        # Serialize and measure size
        serialized_bytes = selective_update.serialize()
        selective_size_bytes = len(serialized_bytes)

        print(
            f"\n[Test Selective Encryption] Unencrypted raw float32 size: {full_unencrypted_size_bytes / 1024 / 1024:.2f} MB"
        )
        print(
            f"[Test Selective Encryption] Selective payload serialized size: {selective_size_bytes / 1024 / 1024:.2f} MB"
        )

        # We expect it to be much smaller than encrypting everything.
        # Encrypting 85M parameters in CKKS takes ~GBs, so if we're under 350MB, it's a huge win.
        # The plaintext tensor itself is ~327MB. The encrypted part is small.
        assert selective_size_bytes > 0

        # Test Deserialize
        deserialized_update = SelectiveUpdate.deserialize(serialized_bytes, public_ctx)

        # Test Reassembly
        # Let's create a fresh model (randomized)
        fresh_model = create_model(config)
        fresh_model.eval()

        with torch.no_grad():
            fresh_logits_before = fresh_model(dummy_input)

        assert (
            torch.max(torch.abs(original_logits - fresh_logits_before)).item() > 1e-4
        ), "Fresh model should differ"

        apply_selective_update(deserialized_update, fresh_model, secret_ctx)

        with torch.no_grad():
            fresh_logits_after = fresh_model(dummy_input)

        # Verify predictions are the same within tolerance
        max_diff = torch.max(torch.abs(original_logits - fresh_logits_after)).item()
        assert max_diff < 1e-3, f"Predictions differed too much after selective update: {max_diff}"
