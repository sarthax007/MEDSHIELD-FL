"""Data splitting for per-hospital train/val/test sets.

Task 19 — Create per-hospital train/val/test splits.

Splits a given hospital's slice manifest into training, validation,
and test sets. To prevent data leakage, splitting is strictly
performed at the patient level.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_hospital_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a hospital partition into train/val/test DataFrames by patient.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        The full manifest DataFrame for a single hospital.
    train_ratio : float
        Proportion of patients assigned to training.
    val_ratio : float
        Proportion of patients assigned to validation.
    test_ratio : float
        Proportion of patients assigned to testing.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The (train_df, val_df, test_df) subsets.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    rng = np.random.default_rng(seed)

    # We determine patient class (tumor vs no-tumor) to roughly stratify
    patient_stats = (
        manifest_df.groupby("patient_id").agg(tumor_slices=("has_tumor", "sum")).reset_index()
    )
    patient_stats["patient_class"] = (patient_stats["tumor_slices"] > 0).astype(int)

    train_pids, val_pids, test_pids = [], [], []

    # Stratify by patient class
    for c in patient_stats["patient_class"].unique():
        class_patients = patient_stats[patient_stats["patient_class"] == c]["patient_id"].values
        rng.shuffle(class_patients)

        n_patients = len(class_patients)
        n_train = int(n_patients * train_ratio)
        n_val = int(n_patients * val_ratio)

        # Ensure at least 1 in train/val/test if possible, but keep it simple here
        # If very few patients, sizes might be 0, which is normal for tiny subsets

        train_pids.extend(class_patients[:n_train])
        val_pids.extend(class_patients[n_train : n_train + n_val])
        test_pids.extend(class_patients[n_train + n_val :])

    train_df = manifest_df[manifest_df["patient_id"].isin(train_pids)].copy()
    val_df = manifest_df[manifest_df["patient_id"].isin(val_pids)].copy()
    test_df = manifest_df[manifest_df["patient_id"].isin(test_pids)].copy()

    return train_df, val_df, test_df


def process_all_clients(
    client_manifests: dict[int, pd.DataFrame],
    output_dir: Path | str | None = None,
    seed: int = 42,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Apply train/val/test splits to all hospital clients.

    Parameters
    ----------
    client_manifests : Dict[int, pd.DataFrame]
        A dictionary mapping client_idx to its full manifest DataFrame.
    output_dir : Path | str | None
        If provided, writes the splits to CSVs and saves a summary table.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    Dict[int, Dict[str, pd.DataFrame]]
        A dictionary mapping client_idx to {"train": df, "val": df, "test": df}.
    """
    results = {}
    summaries = []

    for client_idx, cdf in client_manifests.items():
        train_df, val_df, test_df = create_hospital_splits(cdf, seed=seed)
        results[client_idx] = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        summaries.append(
            {
                "Hospital": f"Client {client_idx}",
                "Train Slices": len(train_df),
                "Val Slices": len(val_df),
                "Test Slices": len(test_df),
                "Total Slices": len(cdf),
            }
        )

    summary_df = pd.DataFrame(summaries)
    logger.info("Split Summary:\n%s", summary_df.to_string(index=False))

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path / "split_summary.csv", index=False)
        for client_idx, splits in results.items():
            splits["train"].to_csv(out_path / f"client_{client_idx}_train.csv", index=False)
            splits["val"].to_csv(out_path / f"client_{client_idx}_val.csv", index=False)
            splits["test"].to_csv(out_path / f"client_{client_idx}_test.csv", index=False)

    return results
