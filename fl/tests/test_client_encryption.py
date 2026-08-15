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
        self.assertEqual(payload_array.dtype, np.dtype("O"))

        # In NumPy, a 0-d object array's item() returns the bytes object
        self.assertIsInstance(payload_array.item(), bytes)

        # Try to deserialize it to ensure it's a selective update bytes
        from shared.medshield.crypto.selective import SelectiveUpdate

        try:
            update = SelectiveUpdate.deserialize(payload_array.item(), context)
            self.assertIsNotNone(update)
            # The dummy model has a critical parameter (head), so we should have encrypted data
            self.assertTrue(len(update.encrypted_critical) > 0)
            # We check the type to make sure it's CKKSVector
            import tenseal as ts

            self.assertIsInstance(update.encrypted_critical[0], ts.CKKSVector)
        except Exception as e:
            self.fail(f"Deserialization failed: {e}")


if __name__ == "__main__":
    unittest.main()
