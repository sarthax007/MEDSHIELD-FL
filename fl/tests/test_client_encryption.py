import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

# Adjust path so imports work if run as a script directly
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shared.medshield.crypto.context import create_ckks_context
from fl.client import MedShieldClient


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(10, 2)
        self.non_critical = nn.Linear(10, 10)

    def forward(self, x):
        return self.head(self.non_critical(x))


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.randn(10), torch.tensor(0)


class TestClientEncryption(unittest.TestCase):
    def test_fit_returns_encrypted_payload(self):
        model = DummyModel()
        trainloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        valloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        device = torch.device("cpu")

        context = create_ckks_context()
        context.make_context_public()  # simulate the public context the client gets

        client = MedShieldClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            public_context=context,
            local_epochs=1,
        )

        # Test fit with empty incoming parameters (assuming first round)
        parameters, num_samples, metrics = client.fit([], {})

        self.assertIsInstance(parameters, list)
        self.assertTrue(len(parameters) == 1)

        # Assert that what is returned is an array of object (bytes) not raw floats
        payload_array = parameters[0]
        self.assertIsInstance(payload_array, np.ndarray)
        self.assertEqual(payload_array.dtype, np.dtype("uint8"))

        # In NumPy, we can convert uint8 back to bytes using tobytes()
        payload_bytes = payload_array.tobytes()
        self.assertIsInstance(payload_bytes, bytes)

        # Try to deserialize it to ensure it's a selective update bytes
        from shared.medshield.crypto.selective import SelectiveUpdate

        try:
            update = SelectiveUpdate.deserialize(payload_bytes, context)
            self.assertIsNotNone(update)
            # The dummy model has a critical parameter (head), so we should have encrypted data
            self.assertTrue(len(update.encrypted_critical) > 0)
            # We check the type to make sure it's CKKSVector
            import tenseal as ts

            self.assertIsInstance(update.encrypted_critical[0], ts.CKKSVector)
        except Exception as e:
            self.fail(f"Deserialization failed: {e}")

    def test_set_encrypted_parameters(self):
        model = DummyModel()
        trainloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        valloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        device = torch.device("cpu")

        # Context has both public and secret keys
        context = create_ckks_context()
        
        # Simulate an encrypted update created by another party (e.g. server/aggregator)
        # First we create an update from the initial model
        from shared.medshield.crypto.selective import create_selective_update
        update = create_selective_update(model, context)
        
        # Now modify the model so we can see if set_parameters brings it back or changes it
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
                
        initial_weights = [p.clone() for p in model.parameters()]

        client = MedShieldClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            public_context=context,
            secret_context=context,  # Using the same context as secret_context for test
            local_epochs=1,
        )

        payload_bytes = update.serialize()
        # Pack into the format expected by get_parameters
        encrypted_params = [np.array(payload_bytes, dtype=object)]

        # Apply encrypted parameters
        client.set_parameters(encrypted_params)

        # Verify weights have changed from initial_weights
        weights_changed = False
        for initial_w, current_w in zip(initial_weights, model.parameters()):
            if not torch.allclose(initial_w, current_w):
                weights_changed = True
                break
                
        self.assertTrue(weights_changed, "Model weights did not change after applying encrypted update.")
        
        # Test sensible predictions
        dummy_input = torch.randn(1, 10)
        output = model(dummy_input)
        self.assertFalse(torch.isnan(output).any(), "Model output contains NaN after decryption.")


if __name__ == "__main__":
    unittest.main()
