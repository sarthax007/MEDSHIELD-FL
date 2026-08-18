import logging
import os
import sys
import flwr.client
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from flwr.common import Scalar
import tenseal as ts

# Ensure shared is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)
from medshield.crypto.selective import create_selective_update, apply_selective_update, SelectiveUpdate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedShieldClient(flwr.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        public_context=None,
        secret_context=None,
        local_epochs: int = 1,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.public_context = public_context
        self.secret_context = secret_context
        self.local_epochs = local_epochs

        # Setup optimizer and loss function (Task 24/25)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays."""
        logger.info("Client: get_parameters called")
        if self.public_context is not None:
            update = create_selective_update(self.model, self.public_context)
            payload_bytes = update.serialize()
            return [np.frombuffer(payload_bytes, dtype=np.uint8)]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load a list of NumPy arrays into the model's state dictionary."""
        if not parameters:
            return
            
        # Check if we have received an encrypted payload
        if len(parameters) == 1 and self.secret_context is not None:
            # We assume it's a serialized SelectiveUpdate
            payload_bytes = parameters[0].tobytes() if parameters[0].dtype != np.dtype("O") else parameters[0].item()
            try:
                update = SelectiveUpdate.deserialize(payload_bytes, self.secret_context)
                apply_selective_update(update, self.model, self.secret_context)
                logger.info("Client: Successfully decrypted and applied global update.")
                return
            except Exception as e:
                logger.warning(f"Client: Failed to deserialize encrypted payload: {e}. Falling back to plaintext.")

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local data."""
        logger.info(f"Client: fit called. Training for {self.local_epochs} epoch(s)...")
        self.set_parameters(parameters)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.local_epochs):
            running_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            logger.info(
                f"Epoch {epoch+1}/{self.local_epochs} - Loss: {running_loss/len(self.trainloader):.4f}"
            )

        train_len = len(self.trainloader.dataset)  # type: ignore
        return self.get_parameters(config={}), train_len, {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation set."""
        logger.info("Client: evaluate called. Evaluating...")
        self.set_parameters(parameters)

        self.model.to(self.device)
        self.model.eval()

        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss / len(self.valloader) if len(self.valloader) > 0 else 0.0

        logger.info(f"Evaluation - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        val_len = len(self.valloader.dataset)  # type: ignore
        return (
            float(avg_loss),
            val_len,
            {"accuracy": float(accuracy)},
        )


def main():
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    logger.info(f"Starting Flower client connecting to {server_address}")

    # Import your actual model and data loaders
    from medshield.model.vit import (
        TumorClassifier,
    )  # Adjust import path if needed based on your structure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TumorClassifier(num_classes=4)

    # Load public context if using encryption
    public_context_path = os.environ.get("HE_PUBLIC_CONTEXT_PATH", "./keys/public.ctx")
    public_context = None
    if os.path.exists(public_context_path):
        with open(public_context_path, "rb") as f:
            public_context = ts.Context.load(f.read())

    # Load secret context for decryption
    secret_context_path = os.environ.get("HE_SECRET_CONTEXT_PATH", "./keys/secret.ctx")
    secret_context = None
    if os.path.exists(secret_context_path):
        with open(secret_context_path, "rb") as f:
            secret_context = ts.Context.load(f.read())

    client_partition = int(os.environ.get("CLIENT_PARTITION_ID", "1"))
    seed = 42 + client_partition
    
    # Set reproducible random seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Placeholder loaders (replace with your actual data partitioning loader)
    # trainloader, valloader = get_dataloaders(...)
    # For now, if you want to test connectivity with dummy data loaders:
    from torch.utils.data import DataLoader, TensorDataset

    # Generate slightly different dummy data per partition
    dummy_data = torch.randn(10, 3, 224, 224) + client_partition
    dummy_labels = torch.randint(0, 4, (10,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    trainloader = DataLoader(dataset, batch_size=2)
    valloader = DataLoader(dataset, batch_size=2)

    client = MedShieldClient(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        public_context=public_context,
        secret_context=secret_context,
        local_epochs=1,
    )

    use_tls = os.environ.get("FL_USE_TLS", "false").lower() == "true"
    root_certificates = None
    if use_tls:
        ca_cert_path = os.environ.get("FL_CA_CERT_PATH", ".certificates/ca.crt")
        from pathlib import Path
        root_certificates = Path(ca_cert_path).read_bytes()
        logger.info("Transport layer security (TLS) is ENABLED.")
    else:
        logger.warning("Transport layer security (TLS) is DISABLED.")

    # Start the Flower client
    flwr.client.start_client(
        server_address=server_address,
        client=client.to_client(),
        root_certificates=root_certificates,
    )
if __name__ == "__main__":
    main()