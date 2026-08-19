from typing import List, Dict, Callable, Optional, Set
import numpy as np
import datetime

from shared.medshield.active.pool import DataPoolManager
from shared.medshield.active.budget import BudgetManager
from shared.medshield.active.uncertainty import UncertaintyStrategy
from shared.medshield.active.query import QueryStrategy

class LabellingQueueService:
    def __init__(
        self,
        pool_manager: DataPoolManager,
        budget_manager: BudgetManager,
        uncertainty_strategy: UncertaintyStrategy,
        query_strategy: QueryStrategy,
        allowed_classes: Optional[Set[int]] = None
    ):
        self.pool = pool_manager
        self.budget = budget_manager
        self.uncertainty = uncertainty_strategy
        self.query = query_strategy
        self.allowed_classes = allowed_classes if allowed_classes is not None else {0, 1, 2, 3}

    def get_labelling_queue(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        features_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        use_diversity: bool = False
    ) -> List[Dict]:
        """
        Retrieves the top candidate images for labelling, up to the remaining budget.
        
        Args:
            predict_fn: A function that takes a list of item_ids and returns 
                        a numpy array of probabilities shape (N, num_classes).
            features_fn: Optional function to get features for diversity calculation.
            use_diversity: Whether to use diversity in the query strategy.
            
        Returns:
            A list of dicts: [{"item_id": str, "prediction": int, "uncertainty": float}]
        """
        unlabeled_ids = self.pool.get_unlabeled_pool()
        if not unlabeled_ids:
            return []
            
        # Check how many we can select
        allowed_k = self.budget.get_allowed_batch_size(len(unlabeled_ids))
        if allowed_k == 0:
            return []

        # Get probabilities for all unlabeled items
        probabilities = predict_fn(unlabeled_ids)
        
        # Calculate uncertainties
        uncertainties = self.uncertainty.calculate_uncertainty(probabilities)
        
        # Get features if needed
        features = None
        if use_diversity and features_fn:
            features = features_fn(unlabeled_ids)
            
        # Select top K
        selected_ids = self.query.select_top_k(
            item_ids=unlabeled_ids,
            uncertainties=uncertainties,
            k=allowed_k,
            features=features,
            use_diversity=use_diversity
        )
        
        # Format the response queue
        queue = []
        for item_id in selected_ids:
            idx = unlabeled_ids.index(item_id)
            pred_class = int(np.argmax(probabilities[idx]))
            score = float(uncertainties[idx])
            
            queue.append({
                "item_id": item_id,
                "prediction": pred_class,
                "uncertainty": score
            })
            
        return queue

    def submit_label(self, item_id: str, class_idx: int, user_id: str) -> bool:
        """
        Validates the submitted label and updates the dataset.
        
        Args:
            item_id: The ID of the image being labelled.
            class_idx: The submitted label.
            user_id: The user submitting the label (for audit log).
            
        Returns:
            True if label was successfully recorded, False otherwise (e.g. invalid).
        """
        if class_idx not in self.allowed_classes:
            raise ValueError(f"Invalid class_idx: {class_idx}. Allowed: {self.allowed_classes}")
            
        if item_id not in self.pool.get_unlabeled_pool():
            raise ValueError(f"item_id {item_id} is not in the unlabeled pool.")
            
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self.pool.submit_label(
            item_id=item_id, 
            label=class_idx, 
            user_id=user_id, 
            timestamp=timestamp
        )
        self.budget.consume_budget(1)
        return True
