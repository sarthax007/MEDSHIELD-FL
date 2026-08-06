"""Tests for the evaluation metrics module.

Task 27 — Implement evaluation metrics.
"""

import numpy as np
import torch

from medshield.model import compute_metrics


def test_compute_metrics_correctness():
    """Test that compute_metrics returns expected values on a known dummy set."""
    # 4 samples, 2 classes
    logits = torch.tensor(
        [
            [2.0, -2.0],  # Pred: 0
            [-2.0, 2.0],  # Pred: 1
            [-2.0, 2.0],  # Pred: 1
            [2.0, -2.0],  # Pred: 0
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])

    # Predictions are: [0, 1, 1, 0]
    # Labels are:      [0, 1, 0, 1]

    # Class 0:
    # True positives: 1 (sample 0)
    # False positives: 1 (sample 3, pred 0 but label 1)
    # False negatives: 1 (sample 2, pred 1 but label 0)
    # Precision 0: 1 / 2 = 0.5
    # Recall 0: 1 / 2 = 0.5
    # F1 0: 0.5

    # Class 1:
    # True positives: 1 (sample 1)
    # False positives: 1 (sample 2, pred 1 but label 0)
    # False negatives: 1 (sample 3, pred 0 but label 1)
    # Precision 1: 1 / 2 = 0.5
    # Recall 1: 1 / 2 = 0.5
    # F1 1: 0.5

    # Macro metrics:
    # Precision: 0.5
    # Recall: 0.5
    # F1: 0.5

    # Accuracy: 2 / 4 = 0.5

    metrics = compute_metrics(logits, labels)

    # 1. Accuracy
    assert np.isclose(metrics["accuracy"], 0.5)

    # 2. Macro metrics
    assert np.isclose(metrics["precision_macro"], 0.5)
    assert np.isclose(metrics["recall_macro"], 0.5)
    assert np.isclose(metrics["f1_macro"], 0.5)

    # 3. Per-class metrics
    assert np.allclose(metrics["precision_per_class"], [0.5, 0.5])
    assert np.allclose(metrics["recall_per_class"], [0.5, 0.5])
    assert np.allclose(metrics["f1_per_class"], [0.5, 0.5])

    # 4. Confusion Matrix
    expected_cm = np.array([[1, 1], [1, 1]])
    assert np.array_equal(metrics["confusion_matrix"], expected_cm)

    # 5. ROC-AUC
    # Expected probs for class 1:
    # Sample 0: sigmoid(-4) ~ 0.0179
    # Sample 1: sigmoid(4) ~ 0.9820
    # Sample 2: sigmoid(4) ~ 0.9820
    # Sample 3: sigmoid(-4) ~ 0.0179
    # Probs: [0.0179, 0.9820, 0.9820, 0.0179]
    # Labels: [0, 1, 0, 1]
    # We can just check it returns a float without NaN since sklearn handles the math.
    assert "roc_auc" in metrics
    assert not np.isnan(metrics["roc_auc"])


def test_compute_metrics_numpy_support():
    """Test that compute_metrics supports numpy arrays."""
    logits = np.array([[2.0, -2.0], [-2.0, 2.0]])
    labels = np.array([0, 1])

    metrics = compute_metrics(logits, labels)
    assert np.isclose(metrics["accuracy"], 1.0)


def test_compute_metrics_single_class_batch():
    """Test that compute_metrics handles a batch with a single true class gracefully."""
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    labels = torch.tensor([0, 0])

    metrics = compute_metrics(logits, labels)
    assert np.isclose(metrics["accuracy"], 1.0)
    # roc_auc is expected to be NaN when there's only one class present
    assert np.isnan(metrics["roc_auc"])
