"""Tests for the single-hospital baseline runner.

Task 31 — Tests for run_baseline.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from medshield.model.config import ModelConfig
from medshield.model.loss import get_loss_function
from medshield.model.run_baseline import (
    evaluate_model,
    generate_results_report,
    generate_synthetic_data,
    seed_everything,
)


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


class TestSeedEverything:
    """Tests for the seed_everything helper."""

    def test_torch_seed_determinism(self) -> None:
        """Verify that seeding produces identical random tensors."""
        seed_everything(123)
        a = torch.randn(5)

        seed_everything(123)
        b = torch.randn(5)

        assert torch.allclose(a, b), "Seeded tensors should be identical."

    def test_numpy_seed_determinism(self) -> None:
        """Verify that seeding produces identical numpy arrays."""
        seed_everything(99)
        a = np.random.rand(5)

        seed_everything(99)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)


class TestGenerateSyntheticData:
    """Tests for the synthetic data generator."""

    def test_generates_manifest_and_files(self, tmp_dir: Path) -> None:
        """Manifest CSV and .npy files are created."""
        manifest_path = generate_synthetic_data(
            data_dir=tmp_dir,
            num_patients=3,
            slices_per_patient=2,
            seed=42,
        )
        assert manifest_path.exists()
        assert manifest_path.name == "manifest.csv"

        # Check .npy files exist
        npy_files = list(tmp_dir.glob("*.npy"))
        assert len(npy_files) == 6  # 3 patients × 2 slices

    def test_manifest_schema(self, tmp_dir: Path) -> None:
        """Manifest has the expected columns."""
        import pandas as pd

        manifest_path = generate_synthetic_data(
            data_dir=tmp_dir,
            num_patients=2,
            slices_per_patient=2,
            seed=42,
        )
        df = pd.read_csv(manifest_path)
        expected_cols = {"filename", "patient_id", "slice_idx", "has_tumor", "tumor_pixels"}
        assert expected_cols.issubset(set(df.columns))

    def test_slice_shape(self, tmp_dir: Path) -> None:
        """Generated slices have the expected shape (4, 240, 240)."""
        generate_synthetic_data(
            data_dir=tmp_dir,
            num_patients=1,
            slices_per_patient=1,
            image_shape=(240, 240),
            num_channels=4,
            seed=42,
        )
        npy_file = list(tmp_dir.glob("*.npy"))[0]
        data = np.load(npy_file)
        assert data.shape == (4, 240, 240)

    def test_deterministic_output(self, tmp_dir: Path) -> None:
        """Same seed produces identical data."""
        dir_a = tmp_dir / "a"
        dir_b = tmp_dir / "b"

        generate_synthetic_data(dir_a, num_patients=2, slices_per_patient=1, seed=77)
        generate_synthetic_data(dir_b, num_patients=2, slices_per_patient=1, seed=77)

        for npy_a in sorted(dir_a.glob("*.npy")):
            npy_b = dir_b / npy_a.name
            np.testing.assert_array_equal(np.load(npy_a), np.load(npy_b))


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_expected_keys(self) -> None:
        """Evaluation returns dict with essential metric keys."""
        # Build a tiny model
        model = torch.nn.Linear(10, 2)
        loss_fn = get_loss_function()
        device = torch.device("cpu")

        # Create a tiny dataset
        data = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = torch.utils.data.TensorDataset(data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        metrics = evaluate_model(model, loader, loss_fn, device)

        assert "accuracy" in metrics
        assert "test_loss" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 8

    def test_empty_loader(self) -> None:
        """Handles an empty DataLoader gracefully."""
        model = torch.nn.Linear(10, 2)
        loss_fn = get_loss_function()
        device = torch.device("cpu")

        dataset = torch.utils.data.TensorDataset(torch.randn(0, 10), torch.randint(0, 2, (0,)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        metrics = evaluate_model(model, loader, loss_fn, device)
        assert "test_loss" in metrics


class TestGenerateResultsReport:
    """Tests for the report generation."""

    def test_creates_markdown_and_json(self, tmp_dir: Path) -> None:
        """Both .md and .json files are written."""
        results = {
            "task": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "seed": 42,
            "training_duration_seconds": 1.5,
            "home_hospital": {
                "test_loss": 0.5,
                "accuracy": 0.75,
                "precision_macro": 0.7,
                "recall_macro": 0.72,
                "f1_macro": 0.71,
                "roc_auc": 0.8,
                "num_samples": 10,
                "confusion_matrix": [[3, 2], [1, 4]],
            },
            "cross_hospital": {
                "Hospital 1": {
                    "accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "roc_auc": 0.5,
                    "num_samples": 10,
                },
            },
        }
        config = ModelConfig()

        md_path, json_path = generate_results_report(results, config, tmp_dir)

        assert md_path.exists()
        assert json_path.exists()
        assert md_path.suffix == ".md"
        assert json_path.suffix == ".json"

    def test_json_is_valid(self, tmp_dir: Path) -> None:
        """JSON file is parsable."""
        results = {
            "task": "test",
            "home_hospital": {"accuracy": 0.9, "test_loss": 0.1},
            "cross_hospital": {},
        }
        config = ModelConfig()

        _, json_path = generate_results_report(results, config, tmp_dir)

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["task"] == "test"

    def test_markdown_contains_key_sections(self, tmp_dir: Path) -> None:
        """Markdown report has all required sections."""
        results = {
            "task": "test",
            "timestamp": "2026-01-01",
            "seed": 42,
            "training_duration_seconds": 10.0,
            "home_hospital": {
                "test_loss": 0.5,
                "accuracy": 0.75,
                "precision_macro": 0.7,
                "recall_macro": 0.72,
                "f1_macro": 0.71,
                "roc_auc": 0.8,
                "num_samples": 10,
            },
            "cross_hospital": {
                "Hospital 1": {
                    "accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "roc_auc": 0.5,
                    "num_samples": 10,
                },
            },
        }
        config = ModelConfig()

        md_path, _ = generate_results_report(results, config, tmp_dir)
        content = md_path.read_text()

        assert "# Single-Hospital Baseline Results" in content
        assert "Model Configuration" in content
        assert "Home Hospital" in content
        assert "Cross-Hospital Generalization" in content
        assert "Reproducibility" in content
        assert "small-data weakness" in content
