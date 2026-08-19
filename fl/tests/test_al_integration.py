import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)

from medshield.active.pool import DataPoolManager
from medshield.active.budget import BudgetManager
from medshield.active.query import QueryStrategy
from medshield.active.uncertainty import LeastConfidenceStrategy
from medshield.active.service import LabellingQueueService
from fl.client.client import MedShieldClient

class DummyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def test_active_learning_simulation_cycle(tmp_path):
    """
    Test that a single cycle of AL accurately updates the pool
    and passes the newly labelled subset to the client.
    """
    # 1. Setup Dummy Data
    num_samples = 20
    features = torch.randn(num_samples, 10)
    true_labels = torch.randint(0, 4, (num_samples,))
    dataset = TensorDataset(features, true_labels)
    
    # 2. Setup AL components
    all_item_ids = [f"item_{i}" for i in range(num_samples)]
    initial_labelled = all_item_ids[:2] # 2 items labelled
    
    pool_file = os.path.join(str(tmp_path), "test_pool_state.json")
    pool_manager = DataPoolManager(
        state_file_path=pool_file,
        initial_items=all_item_ids,
        initial_labelled=initial_labelled
    )
    
    for item_id in initial_labelled:
        idx = int(item_id.split("_")[1])
        pool_manager.submit_label(item_id, int(true_labels[idx].item()), user_id="seeder")
        
    budget_manager = BudgetManager(initial_budget=5)
    query_strategy = QueryStrategy()
    service = LabellingQueueService(
        pool_manager=pool_manager,
        budget_manager=budget_manager,
        uncertainty_strategy=LeastConfidenceStrategy(),
        query_strategy=query_strategy,
        allowed_classes={0, 1, 2, 3}
    )
    
    # Check initial pool sizes
    assert len(pool_manager.get_labelled_pool()) == 2
    assert len(pool_manager.get_unlabeled_pool()) == 18
    
    model = DummyModel()
    
    # 3. Simulate Query
    unlabeled_ids = pool_manager.get_unlabeled_pool()
    predictions_map = {}
    for item_id in unlabeled_ids:
        # Dummy prediction
        predictions_map[item_id] = np.array([0.1, 0.7, 0.1, 0.1])
        
    def mock_predict_fn(item_ids):
        return np.array([predictions_map[item_id] for item_id in item_ids])
        
    queue = service.get_labelling_queue(mock_predict_fn)
    assert len(queue) == 5 # budget limit
    
    # 4. Simulate Labelling
    for item in queue:
        item_id = item["item_id"]
        idx = int(item_id.split("_")[1])
        service.submit_label(item_id, int(true_labels[idx].item()), user_id="test_doc")
        
    assert len(pool_manager.get_labelled_pool()) == 7
    assert len(pool_manager.get_unlabeled_pool()) == 13
    
    # 5. Build dynamic subset for Client
    current_labelled = pool_manager.get_labelled_pool()
    labelled_indices = [int(item_id.split("_")[1]) for item_id in current_labelled]
    
    train_subset = Subset(dataset, labelled_indices)
    trainloader = DataLoader(train_subset, batch_size=2)
    
    assert len(train_subset) == 7
    
    # Validate client can initialize with this trainloader
    client = MedShieldClient(
        model=model,
        trainloader=trainloader,
        valloader=trainloader, # dummy
        device=torch.device("cpu"),
        local_epochs=1
    )
    assert client.trainloader == trainloader
