import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure shared is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)

from medshield.crypto.context import create_ckks_context
from medshield.crypto.selective import SelectiveUpdate

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class EncryptedFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        if "evaluate_metrics_aggregation_fn" not in kwargs:
            kwargs["evaluate_metrics_aggregation_fn"] = self._aggregate_evaluate_metrics
        super().__init__(*args, **kwargs)
        # The server needs a public context to deserialize and perform homomorphic operations.
        self.context = create_ckks_context()
        self.context.make_context_public()

    def _aggregate_evaluate_metrics(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        if not metrics:
            return {}
        total_examples = sum([num_examples for num_examples, _ in metrics])
        if total_examples == 0:
            return {}
        weighted_accuracy = sum([num_examples * float(m.get("accuracy", 0.0)) for num_examples, m in metrics])
        return {"accuracy": weighted_accuracy / total_examples}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using homomorphic encryption."""
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} clients failed to complete fit.")

        if not results:
            return None, {}
            
        participating_clients = [client.cid for client, _ in results]
        logger.info(f"Round {server_round}: Participating clients: {participating_clients}")

        # Store participating clients to use in aggregate_evaluate
        if not hasattr(self, 'round_clients'):
            self.round_clients = {}
        self.round_clients[server_round] = participating_clients

        # Extract updates and sample counts
        updates: List[Tuple[SelectiveUpdate, int]] = []
        plaintext_results = []
        total_examples = 0
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if len(ndarrays) == 1:
                try:
                    if ndarrays[0].dtype == np.dtype("O"):
                        serialized_update = ndarrays[0].item()
                    else:
                        serialized_update = ndarrays[0].tobytes()
                    update = SelectiveUpdate.deserialize(serialized_update, self.context)
                    updates.append((update, fit_res.num_examples))
                    total_examples += fit_res.num_examples
                except Exception as e:
                    logger.warning(f"Failed to deserialize encrypted update: {e}")
            else:
                plaintext_results.append((client, fit_res))

        if plaintext_results and not updates:
            logger.info(f"Aggregating {len(plaintext_results)} plaintext updates via standard FedAvg.")
            agg_parameters, metrics = super().aggregate_fit(server_round, plaintext_results, failures)
            if agg_parameters is not None:
                try:
                    import pickle
                    from fl.db import save_global_model
                    save_global_model(server_round, pickle.dumps(parameters_to_ndarrays(agg_parameters)))
                    logger.info(f"Saved plaintext global model for round {server_round} to database.")
                except Exception as e:
                    logger.error(f"Failed to save plaintext global model to database: {e}")
            return agg_parameters, metrics

        if not updates:
            return None, {}

        if total_examples == 0:
            logger.warning("Total examples is 0 across all clients. Returning None.")
            return None, {}

        logger.info(f"Aggregating encrypted updates from {len(updates)} clients.")

        # Aggregate homomorphically
        first_update, first_examples = updates[0]
        weight = first_examples / total_examples

        agg_encrypted_critical = [
            chunk * weight for chunk in first_update.encrypted_critical
        ]
        agg_plaintext_non_critical = first_update.plaintext_non_critical * weight

        for update, num_examples in updates[1:]:
            weight = num_examples / total_examples
            for i in range(len(agg_encrypted_critical)):
                agg_encrypted_critical[i] += update.encrypted_critical[i] * weight
            agg_plaintext_non_critical += update.plaintext_non_critical * weight

        aggregated_selective_update = SelectiveUpdate(
            encrypted_critical=agg_encrypted_critical,  # type: ignore
            plaintext_non_critical=agg_plaintext_non_critical,
        )

        payload_bytes = aggregated_selective_update.serialize()
        aggregated_ndarrays = [np.frombuffer(payload_bytes, dtype=np.uint8)]
        
        try:
            from fl.db import save_global_model
            save_global_model(server_round, payload_bytes)
            logger.info(f"Saved global model for round {server_round} to database.")
        except Exception as e:
            logger.error(f"Failed to save global model to database: {e}")

        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Save metrics if valid
        if aggregated_loss is not None:
            accuracy = float(aggregated_metrics.get("accuracy", 0.0)) if aggregated_metrics else 0.0
            participating_clients = getattr(self, "round_clients", {}).get(server_round, [])
            
            try:
                from fl.db import save_metrics
                save_metrics(server_round, accuracy, aggregated_loss, participating_clients)
                logger.info(f"Saved metrics for round {server_round} to database.")
            except Exception as e:
                logger.error(f"Failed to save metrics to database: {e}")
                
        return aggregated_loss, aggregated_metrics
