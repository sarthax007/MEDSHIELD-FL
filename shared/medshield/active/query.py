import numpy as np
from typing import List, Optional

class QueryStrategy:
    def __init__(self):
        pass
        
    def select_top_k(
        self,
        item_ids: List[str],
        uncertainties: np.ndarray,
        k: int,
        features: Optional[np.ndarray] = None,
        use_diversity: bool = False
    ) -> List[str]:
        """
        Select the top K informative samples from the given items.
        
        Args:
            item_ids: List of item IDs corresponding to the uncertainties.
            uncertainties: 1D numpy array of uncertainty scores (higher = more uncertain).
            k: Number of items to select.
            features: 2D numpy array of features for diversity selection (num_items, feature_dim).
            use_diversity: Whether to apply a diversity penalty/selection strategy.
            
        Returns:
            List of the selected K item IDs.
        """
        if len(item_ids) == 0 or k <= 0:
            return []
            
        if len(item_ids) != len(uncertainties):
            raise ValueError("Length of item_ids and uncertainties must match.")
            
        k = min(k, len(item_ids))
        
        # Sort indices by uncertainty descending
        sorted_indices = np.argsort(uncertainties)[::-1]
        
        if not use_diversity:
            # Simple top-K by uncertainty
            top_k_indices = sorted_indices[:k]
            return [item_ids[i] for i in top_k_indices]
            
        # Diversity selection using greedy k-center approach on a highly uncertain subset.
        if features is None:
            raise ValueError("Features must be provided when use_diversity is True.")
            
        if len(features) != len(item_ids):
            raise ValueError("Length of features must match length of item_ids.")
            
        # Candidate pool: top 3*k uncertain items to ensure we pick uncertain but diverse ones
        candidate_pool_size = min(3 * k, len(item_ids))
        candidate_indices = sorted_indices[:candidate_pool_size]
        
        selected_indices = []
        
        # Start with the most uncertain item
        selected_indices.append(candidate_indices[0])
        
        if k == 1:
            return [item_ids[selected_indices[0]]]
            
        # Keep track of min distances to the selected set for the remaining candidates
        first_feature = features[candidate_indices[0]]
        candidate_features = features[candidate_indices]
        
        # Compute squared euclidean distances
        min_distances = np.sum((candidate_features - first_feature) ** 2, axis=1)
        
        while len(selected_indices) < k:
            # Mask out already selected items
            for i, c_idx in enumerate(candidate_indices):
                if c_idx in selected_indices:
                    min_distances[i] = -1.0
                    
            farthest_idx = np.argmax(min_distances)
            new_selected_idx = candidate_indices[farthest_idx]
            selected_indices.append(new_selected_idx)
            
            # Update min_distances
            new_feature = features[new_selected_idx]
            new_distances = np.sum((candidate_features - new_feature) ** 2, axis=1)
            min_distances = np.minimum(min_distances, new_distances)
            
        return [item_ids[i] for i in selected_indices]
