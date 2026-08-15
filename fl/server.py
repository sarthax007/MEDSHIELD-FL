import logging
import os
import sys
import flwr.server

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    num_rounds = int(os.environ.get("FL_NUM_ROUNDS", "3"))
    min_clients = int(os.environ.get("FL_MIN_CLIENTS", "2"))

    logger.info(
        f"Starting Flower server on {server_address} with {num_rounds} rounds and minimum {min_clients} clients."
    )

    from fl.strategy.encrypted_fedavg import EncryptedFedAvg

    strategy = EncryptedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    # Start server
    flwr.server.start_server(
        server_address=server_address,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    logger.info(f"Flower server completed {num_rounds} rounds and is shutting down.")


if __name__ == "__main__":
    main()
