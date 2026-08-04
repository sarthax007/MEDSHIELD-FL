"""Tests for the data splitting module.

Task 19 — Create per-hospital train/val/test splits.
"""

import pandas as pd
import pytest

from medshield.data.splitting import create_hospital_splits


@pytest.fixture
def dummy_hospital_manifest() -> pd.DataFrame:
    """Create a dummy manifest with multiple slices per patient."""
    data = []
    # Create 10 patients, each with 2 slices
    for pid in range(10):
        for s_idx in range(2):
            data.append(
                {
                    "patient_id": str(pid),
                    "filename": f"patient_{pid}_slice_{s_idx}.npy",
                    "has_tumor": pid % 2 == 0,  # Even patients have tumors
                }
            )
    return pd.DataFrame(data)


def test_splitting_no_patient_leakage(dummy_hospital_manifest: pd.DataFrame) -> None:
    """Test that train/val/test splits share no patients."""
    train_df, val_df, test_df = create_hospital_splits(
        manifest_df=dummy_hospital_manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    train_pids = set(train_df["patient_id"])
    val_pids = set(val_df["patient_id"])
    test_pids = set(test_df["patient_id"])

    # Ensure sets are disjoint
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)

    # Ensure all patients are accounted for
    all_pids = train_pids | val_pids | test_pids
    assert all_pids == set(dummy_hospital_manifest["patient_id"])


def test_splitting_slice_sum(dummy_hospital_manifest: pd.DataFrame) -> None:
    """Test that the sum of slices across splits equals the total slices."""
    train_df, val_df, test_df = create_hospital_splits(
        manifest_df=dummy_hospital_manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    total_slices = len(train_df) + len(val_df) + len(test_df)
    assert total_slices == len(dummy_hospital_manifest)
