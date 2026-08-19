import numpy as np
import tempfile
import os
import pytest
from shared.medshield.active.pool import DataPoolManager
from shared.medshield.active.budget import BudgetManager
from shared.medshield.active.uncertainty import PredictionEntropyStrategy
from shared.medshield.active.query import QueryStrategy
from shared.medshield.active.service import LabellingQueueService

def test_labelling_queue_service():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        state_path = tmp.name
        
    try:
        initial_items = ["img_1", "img_2", "img_3", "img_4"]
        pool = DataPoolManager(state_path, initial_items=initial_items)
        budget = BudgetManager(2)
        unc_strat = PredictionEntropyStrategy()
        query_strat = QueryStrategy()
        
        service = LabellingQueueService(
            pool, budget, unc_strat, query_strat, allowed_classes={0, 1}
        )
        
        # Mock predict function: img_2 is very uncertain, img_4 is somewhat uncertain
        def mock_predict(item_ids):
            probs = []
            for item in item_ids:
                if item == "img_1": probs.append([0.9, 0.1])
                elif item == "img_2": probs.append([0.5, 0.5])
                elif item == "img_3": probs.append([0.1, 0.9])
                elif item == "img_4": probs.append([0.6, 0.4])
                else: probs.append([1.0, 0.0])
            return np.array(probs)
            
        queue = service.get_labelling_queue(mock_predict)
        
        # Budget is 2, so it should return top 2 uncertain items: img_2, img_4
        assert len(queue) == 2
        assert queue[0]["item_id"] == "img_2"
        assert queue[1]["item_id"] == "img_4"
        
        # Submit labels
        success1 = service.submit_label("img_2", class_idx=1, user_id="doc1")
        assert success1 is True
        success2 = service.submit_label("img_4", class_idx=0, user_id="doc1")
        assert success2 is True
        
        # Invalid label submission
        with pytest.raises(ValueError):
            service.submit_label("img_1", class_idx=99, user_id="doc1")
            
        # Already labelled/Not in unlabeled pool
        with pytest.raises(ValueError):
            service.submit_label("img_2", class_idx=1, user_id="doc1")
        
        # Budget should be consumed
        assert budget.get_remaining_budget() == 0
        
        # New queue should be empty due to budget constraint
        new_queue = service.get_labelling_queue(mock_predict)
        assert len(new_queue) == 0
        
        # Unlabeled pool should be updated
        assert set(pool.get_unlabeled_pool()) == {"img_1", "img_3"}
        
    finally:
        if os.path.exists(state_path):
            os.remove(state_path)
