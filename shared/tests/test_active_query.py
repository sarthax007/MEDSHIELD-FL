import numpy as np
import pytest
from shared.medshield.active.query import QueryStrategy

def test_query_strategy_no_diversity():
    strategy = QueryStrategy()
    item_ids = ["img_1", "img_2", "img_3", "img_4"]
    uncertainties = np.array([0.1, 0.9, 0.4, 0.8])
    
    # Select top 2
    selected = strategy.select_top_k(item_ids, uncertainties, k=2, use_diversity=False)
    
    # Should pick img_2 (0.9) and img_4 (0.8)
    assert selected == ["img_2", "img_4"]

def test_query_strategy_with_diversity():
    strategy = QueryStrategy()
    item_ids = ["img_1", "img_2", "img_3", "img_4", "img_5"]
    
    # 1 and 2 are highly uncertain but very similar in feature space
    # 3 is also highly uncertain but very different from 1 and 2
    # 4 and 5 are less uncertain
    uncertainties = np.array([0.95, 0.94, 0.90, 0.2, 0.1])
    
    features = np.array([
        [1.0, 1.0], # img_1
        [1.0, 1.01],# img_2 (near duplicate of img_1)
        [10.0, 10.0],# img_3 (very diverse from 1/2)
        [5.0, 5.0], # img_4
        [0.0, 0.0]  # img_5
    ])
    
    # If k=2 without diversity, we'd pick img_1 and img_2.
    selected_no_div = strategy.select_top_k(item_ids, uncertainties, k=2, use_diversity=False)
    assert set(selected_no_div) == {"img_1", "img_2"}
    
    # With diversity, we should pick img_1 and img_3 (since img_2 is too close to img_1)
    selected_div = strategy.select_top_k(item_ids, uncertainties, k=2, features=features, use_diversity=True)
    
    # img_1 is picked first (highest uncertainty), then img_3 is picked because it's furthest from img_1
    assert set(selected_div) == {"img_1", "img_3"}

def test_query_strategy_edge_cases():
    strategy = QueryStrategy()
    
    # Empty inputs
    assert strategy.select_top_k([], np.array([]), k=2) == []
    
    # k > len(item_ids)
    item_ids = ["img_1"]
    uncertainties = np.array([0.5])
    assert strategy.select_top_k(item_ids, uncertainties, k=5) == ["img_1"]
    
    # Missing features when diversity is requested
    with pytest.raises(ValueError):
        strategy.select_top_k(item_ids, uncertainties, k=1, use_diversity=True)
