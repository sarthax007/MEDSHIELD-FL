import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
import torch.nn as nn

# Adjust paths to import shared and fl modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from shared.medshield.crypto.context import create_ckks_context
from shared.medshield.crypto.selective import (
    SelectiveUpdate,
    create_selective_update,
    apply_selective_update,
)
from fl.strategy.encrypted_fedavg import EncryptedFedAvg


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 'head' is treated as a critical parameter by the serialization logic
        self.head = nn.Linear(10, 2)
        # 'non_critical' is treated as a non-critical parameter
        self.non_critical = nn.Linear(10, 10)


class TestEncryptedFedAvg(unittest.TestCase):
    def test_aggregate_fit_homomorphic_sum(self):
        # 1. Create a TenSEAL context with both public and secret keys.
        secret_context = create_ckks_context()

        # Create a public-only copy for the server and client encryption
        public_context = secret_context.copy()
        public_context.make_context_public()

        # Initialize the strategy
        strategy = EncryptedFedAvg()

        # Override the server's randomly generated context with our synchronized public context
        # This simulates a real scenario where KMS distributes the same public context to the server
        strategy.context = public_context

        # 2. Create two fake client updates
        model1 = DummyModel()
        with torch.no_grad():
            model1.head.weight.fill_(1.0)
            model1.head.bias.fill_(1.0)
            model1.non_critical.weight.fill_(1.0)
            model1.non_critical.bias.fill_(1.0)

        update1 = create_selective_update(model1, public_context)

        model2 = DummyModel()
        with torch.no_grad():
            model2.head.weight.fill_(3.0)
            model2.head.bias.fill_(3.0)
            model2.non_critical.weight.fill_(3.0)
            model2.non_critical.bias.fill_(3.0)

        update2 = create_selective_update(model2, public_context)

        # Pack them into FitRes messages
        # We assign equal number of examples (10) to both so they have equal weight (0.5 each)
        res1 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(
                [np.frombuffer(update1.serialize(), dtype=np.uint8)]
            ),
            num_examples=10,
            metrics={},
        )
        res2 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(
                [np.frombuffer(update2.serialize(), dtype=np.uint8)]
            ),
            num_examples=10,
            metrics={},
        )

        client_proxy_mock1 = MagicMock(spec=ClientProxy)
        client_proxy_mock2 = MagicMock(spec=ClientProxy)

        results = [(client_proxy_mock1, res1), (client_proxy_mock2, res2)]

        # 3. Pass them through our new EncryptedFedAvg.aggregate_fit() method.
        aggregated_params, _ = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )

        # 4. Assert that the server successfully aggregates them into a single payload WITHOUT needing or using the secret key.
        self.assertIsNotNone(aggregated_params)
        self.assertFalse(strategy.context.has_secret_key())

        # 5. Finally, decrypt the aggregated result using the secret key and assert that it mathematically matches a standard plaintext weighted average (FedAvg) calculation.
        ndarrays = parameters_to_ndarrays(aggregated_params)
        self.assertEqual(len(ndarrays), 1)
        self.assertEqual(ndarrays[0].dtype, np.uint8)

        serialized_update = ndarrays[0].tobytes()
        aggregated_update = SelectiveUpdate.deserialize(
            serialized_update, public_context
        )

        # Apply the update to a blank DummyModel using the secret_context
        aggregated_model = DummyModel()
        apply_selective_update(aggregated_update, aggregated_model, secret_context)

        # The expected average of 1.0 and 3.0 is 2.0 (since weights are equal)
        expected_val = 2.0

        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    aggregated_model.head.weight,
                    torch.tensor(expected_val).expand_as(aggregated_model.head.weight),
                    atol=1e-3,
                )
            )
            self.assertTrue(
                torch.allclose(
                    aggregated_model.head.bias,
                    torch.tensor(expected_val).expand_as(aggregated_model.head.bias),
                    atol=1e-3,
                )
            )
            self.assertTrue(
                torch.allclose(
                    aggregated_model.non_critical.weight,
                    torch.tensor(expected_val).expand_as(
                        aggregated_model.non_critical.weight
                    ),
                    atol=1e-3,
                )
            )
            self.assertTrue(
                torch.allclose(
                    aggregated_model.non_critical.bias,
                    torch.tensor(expected_val).expand_as(
                        aggregated_model.non_critical.bias
                    ),
                    atol=1e-3,
                )
            )

    def test_aggregate_fit_weighted_sum(self):
        secret_context = create_ckks_context()
        public_context = secret_context.copy()
        public_context.make_context_public()
        strategy = EncryptedFedAvg()
        strategy.context = public_context

        model1 = DummyModel()
        with torch.no_grad():
            model1.head.weight.fill_(1.0)
            model1.head.bias.fill_(1.0)
            model1.non_critical.weight.fill_(1.0)
            model1.non_critical.bias.fill_(1.0)

        update1 = create_selective_update(model1, public_context)

        model2 = DummyModel()
        with torch.no_grad():
            model2.head.weight.fill_(5.0)
            model2.head.bias.fill_(5.0)
            model2.non_critical.weight.fill_(5.0)
            model2.non_critical.bias.fill_(5.0)

        update2 = create_selective_update(model2, public_context)

        # 10 examples for model1 (1.0), 30 examples for model2 (5.0)
        # Expected average = (1.0 * 10 + 5.0 * 30) / 40 = 160 / 40 = 4.0
        res1 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.frombuffer(update1.serialize(), dtype=np.uint8)]),
            num_examples=10,
            metrics={},
        )
        res2 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.frombuffer(update2.serialize(), dtype=np.uint8)]),
            num_examples=30,
            metrics={},
        )

        results = [(MagicMock(spec=ClientProxy), res1), (MagicMock(spec=ClientProxy), res2)]
        aggregated_params, _ = strategy.aggregate_fit(1, results, [])
        assert aggregated_params is not None

        ndarrays = parameters_to_ndarrays(aggregated_params)
        aggregated_update = SelectiveUpdate.deserialize(ndarrays[0].tobytes(), public_context)
        
        aggregated_model = DummyModel()
        apply_selective_update(aggregated_update, aggregated_model, secret_context)

        expected_val = 4.0
        with torch.no_grad():
            self.assertTrue(torch.allclose(aggregated_model.head.weight, torch.tensor(expected_val).expand_as(aggregated_model.head.weight), atol=1e-3))
            self.assertTrue(torch.allclose(aggregated_model.non_critical.weight, torch.tensor(expected_val).expand_as(aggregated_model.non_critical.weight), atol=1e-3))

    def test_aggregate_fit_zero_examples(self):
        secret_context = create_ckks_context()
        public_context = secret_context.copy()
        public_context.make_context_public()
        strategy = EncryptedFedAvg()
        strategy.context = public_context
        
        model = DummyModel()
        update = create_selective_update(model, public_context)
        payload_bytes = update.serialize()
        
        res1 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.frombuffer(payload_bytes, dtype=np.uint8)]),
            num_examples=0,
            metrics={},
        )
        res2 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.frombuffer(payload_bytes, dtype=np.uint8)]),
            num_examples=0,
            metrics={},
        )

        results = [(MagicMock(spec=ClientProxy), res1), (MagicMock(spec=ClientProxy), res2)]
        
        aggregated_params, _ = strategy.aggregate_fit(1, results, [])
        self.assertIsNone(aggregated_params)


if __name__ == "__main__":
    unittest.main()
