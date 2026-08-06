"""Evaluation metrics computation for the classification model.

Task 27 — Implement evaluation metrics.
"""

from typing import Any, Dict, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    logits: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray]
) -> Dict[str, Any]:
    """Compute classification metrics from model logits and true labels.

    Computes accuracy, precision (macro & per-class), recall (macro & per-class),
    F1 (macro & per-class), ROC-AUC (macro, via ovr), and confusion matrix.

    Parameters
    ----------
    logits : Union[torch.Tensor, np.ndarray]
        The raw logit outputs from the model. Shape: (N, C)
    labels : Union[torch.Tensor, np.ndarray]
        The true class labels. Shape: (N,)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the computed metrics.
    """
    if isinstance(logits, torch.Tensor):
        # Convert logits to probabilities
        probs_np = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds_np = torch.argmax(logits, dim=1).detach().cpu().numpy()
    else:
        # Assuming numpy array input
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_np = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds_np = np.argmax(logits, axis=1)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    metrics: Dict[str, Any] = {}
    
    # 1. Accuracy
    metrics["accuracy"] = float(accuracy_score(labels_np, preds_np))
    
    # 2. Precision, Recall, F1 (Macro)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="macro", zero_division="warn"  # type: ignore
    )
    metrics["precision_macro"] = float(prec_macro)
    metrics["recall_macro"] = float(rec_macro)
    metrics["f1_macro"] = float(f1_macro)
    
    # 3. Precision, Recall, F1 (Per Class)
    prec_class, rec_class, f1_class, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None, zero_division="warn"  # type: ignore
    )
    metrics["precision_per_class"] = prec_class
    metrics["recall_per_class"] = rec_class
    metrics["f1_per_class"] = f1_class
    
    # 4. Confusion Matrix
    metrics["confusion_matrix"] = confusion_matrix(labels_np, preds_np)
    
    # 5. ROC-AUC (Macro, One-vs-Rest)
    # Note: For multi-class, ROC-AUC needs probabilities for all classes
    # If the batch only contains one class (can happen in small sets), roc_auc_score throws an error.
    try:
        # Use multi_class="ovr" as the standard approach
        if probs_np.shape[1] == 2:
            # For binary classification, roc_auc_score expects probabilities of the greater label
            roc_auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            roc_auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
        metrics["roc_auc"] = float(roc_auc)
    except ValueError:
        # E.g., Only one class present in y_true
        metrics["roc_auc"] = float("nan")

    return metrics
