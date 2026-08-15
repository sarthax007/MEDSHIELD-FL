import logging
import os
import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedShieldClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        public_context=None,
        local_epochs: int = 1,
        learning_rate: float = 1e-4
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.public_context = public_context
        self.local_epochs = local_epochs
        
        # Setup optimizer and loss function (Task 24/25)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays."""
        logger.info("Client: get_parameters called")
        if self.public_context is not None:
            import sys
            import os
            # Ensure shared is in path
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared')))
            from medshield.crypto.selective import create_selective_update
            update = create_selective_update(self.model, self.public_context)
            payload_bytes = update.serialize()
            return [np.array(payload_bytes, dtype=object)]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load a list of NumPy arrays into the model's state dictionary."""
        if not parameters:
            return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
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
            
            logger.info(f"Epoch {epoch+1}/{self.local_epochs} - Loss: {running_loss/len(self.trainloader):.4f}")
            
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
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
        
        return float(avg_loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def main():
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    logger.info(f"Starting Flower client connecting to {server_address}")
    
    # Placeholder for loading actual data and model
    # To run this properly, you need the Dataset/DataLoader from Task 18 
    # and the ViT model from Task 21/22.
    # For now, we will use a dummy model and data loader if they are missing,
    # or you can import them from `shared/model` and `data/` modules once built.
    
    # Example (mocked):
    # from shared.model import ViTClassifier
    # from data.loader import get_dataloaders
    # model = ViTClassifier(num_classes=4)
    # trainloader, valloader = get_dataloaders(hospital_id="H1")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # client = MedShieldClient(model, trainloader, valloader, device)
    
    # We will raise NotImplementedError temporarily for the real setup 
    # so we don't start training on nothing if run blindly.
    
    # fl.client.start_client(
    #     server_address=server_address,
    #     client=client.to_client(),
    # )
    
    logger.info("Flower client initialized (but waiting on data/model imports to connect).")


if __name__ == "__main__":
    main()
