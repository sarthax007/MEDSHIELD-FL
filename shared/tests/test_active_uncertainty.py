import numpy as np
import pytest
from shared.medshield.active.uncertainty import (
    PredictionEntropyStrategy,
    LeastConfidenceStrategy,
    MCDropoutStrategy
)

def test_prediction_entropy_strategy():
    strategy = PredictionEntropyStrategy()
    
    # Highly confident prediction (e.g., [0.99, 0.01])
    confident_probs = np.array([[0.99, 0.01]])
    confident_entropy = strategy.calculate_uncertainty(confident_probs)
    
    # Highly uncertain prediction (e.g., [0.5, 0.5])
    uncertain_probs = np.array([[0.5, 0.5]])
    uncertain_entropy = strategy.calculate_uncertainty(uncertain_probs)
    
    # Entropy should be higher for the uncertain prediction
    assert uncertain_entropy[0] > confident_entropy[0]

def test_least_confidence_strategy():
    strategy = LeastConfidenceStrategy()
    
    confident_probs = np.array([[0.9, 0.1]])
    confident_unc = strategy.calculate_uncertainty(confident_probs)
    assert np.isclose(confident_unc[0], 0.1)
    
    uncertain_probs = np.array([[0.5, 0.5]])
    uncertain_unc = strategy.calculate_uncertainty(uncertain_probs)
    assert np.isclose(uncertain_unc[0], 0.5)

def test_mcdropout_strategy():
    strategy = MCDropoutStrategy(num_forward_passes=5)
    
    # Shape: (batch_size, num_passes, num_classes)
    # Batch size 1, 2 passes, 2 classes
    probs = np.array([
        [
            [0.8, 0.2],
            [0.7, 0.3]
        ]
    ])
    
    uncertainty = strategy.calculate_uncertainty(probs)
    assert uncertainty.shape == (1,)
    
    with pytest.raises(ValueError):
        strategy.calculate_uncertainty(np.array([[0.8, 0.2]]))
