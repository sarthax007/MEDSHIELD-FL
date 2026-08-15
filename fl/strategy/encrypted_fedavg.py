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
        super().__init__(*args, **kwargs)
        # The server needs a public context to deserialize and perform homomorphic operations.
        self.context = create_ckks_context()
        self.context.make_context_public()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using homomorphic encryption."""
        if not results:
            return None, {}

        # Extract updates and sample counts
        updates: List[Tuple[SelectiveUpdate, int]] = []
        total_examples = 0
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if len(ndarrays) == 1 and ndarrays[0].dtype == np.uint8:
                serialized_update = ndarrays[0].tobytes()
                update = SelectiveUpdate.deserialize(serialized_update, self.context)
                updates.append((update, fit_res.num_examples))
                total_examples += fit_res.num_examples
            else:
                logger.warning("Received unexpected parameters format from client.")

        if not updates:
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

        return ndarrays_to_parameters(aggregated_ndarrays), {}
