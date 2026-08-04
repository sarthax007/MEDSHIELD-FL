"""Loss function and class-imbalance handling.

Task 24 — Implement the loss function and class-imbalance handling.

Rationale for Imbalance Handling:
The BraTS dataset is heavily imbalanced towards healthy tissue and High-Grade Glioma (HGG).
Instead of using oversampling/undersampling techniques (like SMOTE or RandomOversampler)
which can destabilize federated learning by altering the natural distribution of data on
local clients, we use a weighted Cross-Entropy Loss. This ensures that the model learns to
penalize errors on the minority class more heavily without changing the raw data feed,
keeping federated averaging (FedAvg) more stable.
"""

from typing import Optional

import torch
import torch.nn as nn


def get_loss_function(
    class_counts: Optional[dict[int, int]] = None, device: Optional[torch.device] = None
) -> nn.Module:
    """Build the training loss function, handling class imbalance.

    Parameters
    ----------
    class_counts : Optional[Dict[int, int]], default=None
        A dictionary mapping class indices to their sample counts.
        If provided, the function computes inverse-frequency weights.
    device : Optional[torch.device], default=None
        The device to place the weight tensor on.

    Returns
    -------
    nn.Module
        The PyTorch loss function (e.g., nn.CrossEntropyLoss) ready for training batches.
    """
    weights = None

    if class_counts and len(class_counts) > 0:
        # Sort class indices to ensure consistent weight ordering
        num_classes = max(class_counts.keys()) + 1

        # Calculate inverse frequency weights
        # weight_i = total_samples / (num_classes * count_i)
        total_samples = sum(class_counts.values())

        weight_list = []
        for i in range(num_classes):
            count = class_counts.get(i, 0)
            if count == 0:
                # Assign a nominal weight to avoid division by zero or log issues,
                # or typically 0 so the class doesn't artificially inflate loss if missing
                weight_list.append(0.0)
            else:
                weight_list.append(total_samples / (num_classes * count))

        weights = torch.tensor(weight_list, dtype=torch.float32)
        if device is not None:
            weights = weights.to(device)

    return nn.CrossEntropyLoss(weight=weights)
