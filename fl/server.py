import logging
import os
import flwr as fl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    
    logger.info(f"Starting Flower server on {server_address}")
    
    # Configure a basic FedAvg strategy for 1 round, requiring 1 client
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
    logger.info("Flower server round completed and shutting down.")

if __name__ == "__main__":
    main()
