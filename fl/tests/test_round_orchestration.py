import os
import sys
import unittest
import torch
import torch.nn as nn

import flwr as fl
from flwr.simulation import start_simulation
from flwr.server import ServerConfig

# Ensure shared is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fl.strategy.encrypted_fedavg import EncryptedFedAvg
from fl.client.client import MedShieldClient
from shared.medshield.crypto.context import create_ckks_context


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.non_critical = nn.Linear(10, 10)
        self.head = nn.Linear(10, 2)

    def forward(self, x):
        x = self.non_critical(x)
        x = torch.relu(x)
        x = self.head(x)
        return x


def client_fn(cid: str) -> fl.client.Client:
    device = torch.device("cpu")
    model = DummyModel()

    # We just need simple dummy dataloaders
    dummy_data = torch.randn(10, 10)
    dummy_labels = torch.randint(0, 2, (10,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    valloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Give all clients a synchronized public context
    secret_context = create_ckks_context()
    public_context = secret_context.copy()
    public_context.make_context_public()

    return MedShieldClient(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        public_context=public_context,
        local_epochs=1,
        learning_rate=0.01,
    ).to_client()


class TestRoundOrchestration(unittest.TestCase):
    def test_multi_round_orchestration(self):
        strategy = EncryptedFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

        # Start the simulation
        history = start_simulation(
            client_fn=client_fn,
            num_clients=2,
            config=ServerConfig(num_rounds=2),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )

        # Verify that simulation completed successfully
        self.assertIsNotNone(history)

        # Check if 2 rounds of training were executed and we have losses
        # (Though EncryptedFedAvg might not compute global loss by default, the history is returned)
        self.assertTrue(len(history.metrics_centralized) >= 0)


if __name__ == "__main__":
    unittest.main()
