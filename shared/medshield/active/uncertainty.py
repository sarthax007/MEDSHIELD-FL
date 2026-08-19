import numpy as np
from abc import ABC, abstractmethod

class UncertaintyStrategy(ABC):
    @abstractmethod
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty scores for a batch of predictions.
        
        Args:
            probabilities: A numpy array of shape (batch_size, num_classes) containing probabilities.
            
        Returns:
            A numpy array of shape (batch_size,) containing uncertainty scores.
        """
        pass

class PredictionEntropyStrategy(UncertaintyStrategy):
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate prediction entropy.
        H(x) = - sum(p * log(p))
        Higher entropy means higher uncertainty.
        """
        epsilon = 1e-10
        probs_safe = np.clip(probabilities, epsilon, 1.0)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
        return entropy

class LeastConfidenceStrategy(UncertaintyStrategy):
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate least confidence uncertainty.
        U(x) = 1 - max(p)
        Higher value means higher uncertainty.
        """
        max_probs = np.max(probabilities, axis=1)
        return 1.0 - max_probs

class MCDropoutStrategy(UncertaintyStrategy):
    def __init__(self, num_forward_passes: int = 10):
        self.num_forward_passes = num_forward_passes
        
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Basic structure for MC-Dropout.
        Expects input probabilities of shape (batch_size, num_passes, num_classes).
        """
        if probabilities.ndim == 3:
            mean_probs = np.mean(probabilities, axis=1)
            epsilon = 1e-10
            probs_safe = np.clip(mean_probs, epsilon, 1.0)
            entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
            return entropy
        else:
            raise ValueError("MCDropoutStrategy requires probabilities of shape (batch_size, num_passes, num_classes)")
