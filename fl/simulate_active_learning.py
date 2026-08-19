import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import logging
import json

# Ensure shared is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared"))
)

from medshield.active.pool import DataPoolManager
from medshield.active.budget import BudgetManager
from medshield.active.query import QueryStrategy
from medshield.active.uncertainty import LeastConfidenceStrategy
from medshield.active.service import LabellingQueueService
from fl.client.client import MedShieldClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AL_Simulation")

class DummyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def simulate_active_learning():
    # 1. Setup Dummy Data (Ground Truth)
    num_samples = 100
    num_classes = 4
    features = torch.randn(num_samples, 10)
    true_labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(features, true_labels)
    
    # Validation dataset
    val_features = torch.randn(20, 10)
    val_labels = torch.randint(0, num_classes, (20,))
    val_dataset = TensorDataset(val_features, val_labels)
    valloader = DataLoader(val_dataset, batch_size=4)

    # 2. Setup Active Learning Components
    hospital_id = "sim_hospital"
    data_dir = "./sim_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Delete old state if exists
    pool_file = os.path.join(data_dir, f"{hospital_id}_pool_state.json")
    if os.path.exists(pool_file):
        os.remove(pool_file)
        
    all_item_ids = [f"item_{i}" for i in range(num_samples)]
    # Start with only 5 labelled items
    initial_labelled = all_item_ids[:5]
    
    pool_manager = DataPoolManager(
        state_file_path=pool_file,
        initial_items=all_item_ids,
        initial_labelled=initial_labelled
    )
    
    # We must seed the true labels for the initially labelled items
    for item_id in initial_labelled:
        idx = int(item_id.split("_")[1])
        # Force label submission without validation to seed
        pool_manager.submit_label(item_id, int(true_labels[idx].item()), user_id="seeder")
    
    # 10 labels allowed per round
    budget_manager = BudgetManager(initial_budget=10)
    uncertainty_strategy = LeastConfidenceStrategy()
    query_strategy = QueryStrategy()
    
    service = LabellingQueueService(
        pool_manager=pool_manager,
        budget_manager=budget_manager,
        uncertainty_strategy=uncertainty_strategy,
        query_strategy=query_strategy,
        allowed_classes={0, 1, 2, 3}
    )
    
    # 3. Initialize Model
    device = torch.device("cpu")
    model = DummyModel(num_classes=num_classes)
    
    # 4. Simulation Loop
    num_cycles = 3
    logger.info(f"Starting Active Learning Loop for {num_cycles} cycles...")
    
    for cycle in range(num_cycles):
        logger.info(f"--- Cycle {cycle + 1} ---")
        
        # Step A: Dynamically build DataLoader for current labelled pool
        current_labelled = pool_manager.get_labelled_pool()
        logger.info(f"Current Labelled Pool Size: {len(current_labelled)}")
        
        # Map item_ids back to indices
        labelled_indices = [int(item_id.split("_")[1]) for item_id in current_labelled]
        
        # Build dynamic subset
        train_subset = Subset(dataset, labelled_indices)
        trainloader = DataLoader(train_subset, batch_size=4, shuffle=True)
        
        # Step B: Train Model Locally (simulate client.fit() inside a federated round)
        logger.info("Running local training (simulate federated round fit)...")
        client = MedShieldClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            local_epochs=2,
            learning_rate=0.01
        )
        # Call fit - parameters are just the model state dict here
        parameters = client.get_parameters(config={})
        updated_params, num_examples, _ = client.fit(parameters, config={})
        
        # Evaluate
        loss, val_len, metrics = client.evaluate(updated_params, config={})
        
        # Step C: Active Learning Query
        logger.info("Querying for next batch of labels...")
        
        unlabeled_ids = pool_manager.get_unlabeled_pool()
        if not unlabeled_ids:
            logger.info("No more unlabeled data.")
            break
            
        unlabeled_indices = [int(item_id.split("_")[1]) for item_id in unlabeled_ids]
        
        # Compute probabilities for all unlabeled items
        model.eval()
        unlabeled_subset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=len(unlabeled_subset), shuffle=False)
        with torch.no_grad():
            batch_features, _ = next(iter(unlabeled_loader))
            logits = model(batch_features)
            probs = torch.softmax(logits, dim=1).numpy()
        
        predictions_map = {item_id: probs[i] for i, item_id in enumerate(unlabeled_ids)}
        
        def mock_predict_fn(item_ids):
            return np.array([predictions_map[item_id] for item_id in item_ids])
            
        # Get queue
        budget_manager.reset_budget() # new round, new budget
        queue = service.get_labelling_queue(mock_predict_fn)
        logger.info(f"Received {len(queue)} items for labelling.")
        
        # Step D: Simulate Doctor Labelling
        logger.info("Simulating doctor labels...")
        for item in queue:
            item_id = item["item_id"]
            idx = int(item_id.split("_")[1])
            true_label = int(true_labels[idx].item())
            service.submit_label(item_id, true_label, user_id="sim_doctor")
            
        logger.info(f"Finished Cycle {cycle + 1}. Labelled pool is now: {len(pool_manager.get_labelled_pool())}")
        
    logger.info("Active Learning Simulation Complete!")

if __name__ == "__main__":
    simulate_active_learning()
