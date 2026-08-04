"""Tumor-class labelling map.

Task 15 — Implement the tumor-class labelling map.

Provides a single source-of-truth mapping from raw annotations
(e.g., `has_tumor` boolean or string labels) to model class indices.
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum

logger = logging.getLogger(__name__)


class TumorClass(Enum):
    """Target classification labels for the model."""

    NO_TUMOR = 0
    TUMOR = 1


# Single source-of-truth mapping
LABEL_MAP = {
    "no-tumor": TumorClass.NO_TUMOR.value,
    "tumor": TumorClass.TUMOR.value,
    False: TumorClass.NO_TUMOR.value,
    True: TumorClass.TUMOR.value,
    0: TumorClass.NO_TUMOR.value,
    1: TumorClass.TUMOR.value,
}


def map_raw_label(raw_label: str | bool | int) -> int:
    """Map a raw annotation to a model class index.

    Parameters
    ----------
    raw_label : str | bool | int
        The raw label from the dataset manifest (e.g., `True`, `False`, `"tumor"`).

    Returns
    -------
    int
        The integer class index for the model.

    Raises
    ------
    ValueError
        If the raw label cannot be mapped to any known class.
    """
    if isinstance(raw_label, str):
        raw_label = raw_label.lower().strip()

    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]

    raise ValueError(f"Unmapped or null label encountered: {raw_label}")


def confirm_mapping(raw_labels: list[str | bool | int]) -> dict[int, int]:
    """Produce a count per final class to confirm the mapping is correct.

    Parameters
    ----------
    raw_labels : list[str | bool | int]
        A list of raw labels to map and count.

    Returns
    -------
    dict[int, int]
        A dictionary mapping the final class index to its count.
    """
    mapped_labels = [map_raw_label(lbl) for lbl in raw_labels]
    counts = dict(Counter(mapped_labels))

    # Log the counts for confirmation
    logger.info("Class mapping counts:")
    for class_idx, count in counts.items():
        class_name = TumorClass(class_idx).name
        logger.info(" - %s (Index %d): %d samples", class_name, class_idx, count)

    return counts
