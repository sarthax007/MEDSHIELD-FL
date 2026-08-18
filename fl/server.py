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
    min_fit_clients = int(os.environ.get("FL_MIN_FIT_CLIENTS", "2"))

    logger.info(
        f"Starting Flower server on {server_address} with {num_rounds} rounds, {min_clients} available clients, and minimum {min_fit_clients} fit clients."
    )

    from fl.strategy.encrypted_fedavg import EncryptedFedAvg

    strategy = EncryptedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_clients,
    )

    # Security Documentation:
    # Client-server communication here uses TLS/secure gRPC when configured.
    # This acts as a second, independent layer of defense, securing the transport layer.
    # It is completely separate from the homomorphic encryption layer applied to the model updates themselves.
    
    use_tls = os.environ.get("FL_USE_TLS", "false").lower() == "true"
    certificates = None
    if use_tls:
        ca_cert_path = os.environ.get("FL_CA_CERT_PATH", ".certificates/ca.crt")
        server_cert_path = os.environ.get("FL_SERVER_CERT_PATH", ".certificates/server.pem")
        server_key_path = os.environ.get("FL_SERVER_KEY_PATH", ".certificates/server.key")
        
        from pathlib import Path
        certificates = (
            Path(ca_cert_path).read_bytes(),
            Path(server_cert_path).read_bytes(),
            Path(server_key_path).read_bytes(),
        )
        logger.info("Transport layer security (TLS) is ENABLED.")
    else:
        logger.warning("Transport layer security (TLS) is DISABLED.")

    # Start server
    flwr.server.start_server(
        server_address=server_address,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        certificates=certificates,
    )
    logger.info(f"Flower server completed {num_rounds} rounds and is shutting down.")


if __name__ == "__main__":
    main()
