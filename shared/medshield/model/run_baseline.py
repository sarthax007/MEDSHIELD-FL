"""Single-hospital baseline training and evaluation.

Task 31 — Train and record the single-hospital baseline.

Trains the ViT on one hospital's data with a fixed seed, evaluates on its
own test set and on other hospitals' test sets to demonstrate the
"small-data" weakness, then records all metrics to ``docs/results/``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medshield.data.dataset import create_dataloaders
from medshield.data.labels import confirm_mapping
from medshield.data.partitioning import partition_by_patient
from medshield.data.splitting import create_hospital_splits
from medshield.model.checkpoint import save_checkpoint
from medshield.model.config import ModelConfig
from medshield.model.logging import ExperimentLogger
from medshield.model.loss import get_loss_function
from medshield.model.metrics import compute_metrics
from medshield.model.optimizer import get_optimizer, get_scheduler
from medshield.model.registry import create_model
from medshield.model.train import get_device, train_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (shared/medshield/model/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic_baseline"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "baseline"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "logs" / "baseline"

# ---------------------------------------------------------------------------
# Fixed seed
# ---------------------------------------------------------------------------
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    logger.info("Set all random seeds to %d", seed)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    data_dir: Path,
    num_patients: int = 30,
    slices_per_patient: int = 5,
    image_shape: tuple[int, int] = (240, 240),
    num_channels: int = 4,
    seed: int = SEED,
) -> Path:
    """Generate synthetic MRI slice data that mimics the BraTS format.

    Creates ``.npy`` files with shape ``(C, H, W)`` and a ``manifest.csv``
    in the same format produced by ``slicer.py``.

    Parameters
    ----------
    data_dir : Path
        Directory to write synthetic slices and manifest into.
    num_patients : int
        Number of synthetic patients to create.
    slices_per_patient : int
        Number of slices per patient.
    image_shape : tuple[int, int]
        Spatial dimensions ``(H, W)`` of each slice.
    num_channels : int
        Number of MRI channels (default 4: T1, T1ce, T2, FLAIR).
    seed : int
        Random seed for deterministic generation.

    Returns
    -------
    Path
        Path to the generated ``manifest.csv``.
    """
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for patient_idx in range(1, num_patients + 1):
        # Roughly 60% of patients have tumors
        patient_has_tumor = rng.random() < 0.6

        for slice_idx in range(1, slices_per_patient + 1):
            filename = f"{patient_idx}_slice{slice_idx}.npy"

            # Create base brain-like image (smooth Gaussian blob)
            slice_data = np.zeros((num_channels, image_shape[0], image_shape[1]), dtype=np.float32)

            for ch in range(num_channels):
                # Each channel gets slightly different intensity patterns
                base = rng.standard_normal(image_shape).astype(np.float32) * 50.0
                # Add a central "brain" region
                y, x = np.ogrid[
                    -image_shape[0] // 2 : image_shape[0] // 2,
                    -image_shape[1] // 2 : image_shape[1] // 2,
                ]
                brain_mask = (x * x + y * y) < (80 + ch * 5) ** 2
                base[brain_mask] += 200.0 + ch * 30.0
                slice_data[ch] = base

            # Determine if this particular slice has tumor
            has_tumor = False
            tumor_pixels = 0

            if patient_has_tumor and slice_idx >= 2:
                has_tumor = True
                # Add a bright "tumor" blob offset from centre
                ty = rng.integers(40, 120)
                tx = rng.integers(40, 120)
                tr = rng.integers(10, 30)
                yy, xx = np.ogrid[
                    -image_shape[0] // 2 : image_shape[0] // 2,
                    -image_shape[1] // 2 : image_shape[1] // 2,
                ]
                tumor_mask = (
                    (yy - ty + image_shape[0] // 2) ** 2 + (xx - tx + image_shape[1] // 2) ** 2
                ) < tr**2
                tumor_pixels = int(tumor_mask.sum())
                for ch in range(num_channels):
                    slice_data[ch][tumor_mask] += 300.0 + ch * 50.0

            np.save(data_dir / filename, slice_data)

            rows.append(
                {
                    "filename": filename,
                    "patient_id": patient_idx,
                    "slice_idx": slice_idx,
                    "has_tumor": has_tumor,
                    "tumor_pixels": tumor_pixels,
                }
            )

    manifest_df = pd.DataFrame(rows)
    manifest_path = data_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    logger.info(
        "Generated synthetic data: %d patients, %d slices at %s",
        num_patients,
        len(rows),
        data_dir,
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a trained model on a test DataLoader.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    test_loader : DataLoader
        DataLoader for the test split.
    loss_fn : nn.Module
        The loss function (for computing test loss).
    device : torch.device
        Device to run evaluation on.

    Returns
    -------
    dict[str, Any]
        Dictionary of evaluation metrics including accuracy, precision,
        recall, F1, ROC-AUC, confusion_matrix, and test_loss.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_logits.append(outputs.cpu())
            all_labels.append(targets.cpu())

    if total_samples == 0:
        logger.warning("No samples in test loader — returning empty metrics.")
        return {"test_loss": float("nan"), "accuracy": float("nan")}

    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_logits_tensor, all_labels_tensor)

    # Convert numpy arrays to lists for JSON serialisation
    for key, value in metrics.items():
        if hasattr(value, "tolist"):
            metrics[key] = value.tolist()

    metrics["test_loss"] = total_loss / max(1, total_samples)
    metrics["num_samples"] = total_samples

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_results_report(
    results: dict[str, Any],
    config: ModelConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the baseline results as Markdown and JSON to docs/results/.

    Parameters
    ----------
    results : dict[str, Any]
        Full results dictionary (home hospital + cross-hospital).
    config : ModelConfig
        The model configuration used.
    output_dir : Path
        Directory to write reports into.

    Returns
    -------
    tuple[Path, Path]
        Paths to (markdown_report, json_results).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = output_dir / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info("Wrote JSON results to %s", json_path)

    # --- Markdown ---
    md_path = output_dir / "single_hospital_baseline.md"

    home = results["home_hospital"]
    cross = results.get("cross_hospital", {})
    cfg = config.to_dict()
    ts = results.get("timestamp", "N/A")
    duration = results.get("training_duration_seconds", "N/A")
    seed_val = results.get("seed", SEED)

    lines = [
        "# Single-Hospital Baseline Results",
        "",
        f"**Date**: {ts}  ",
        f"**Seed**: {seed_val}  ",
        f"**Training duration**: {duration}s  ",
        "",
        "## Model Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in cfg.items():
        lines.append(f"| {k} | `{v}` |")

    def _fmt(val: Any, fmt: str = ".4f") -> str:
        """Format a numeric value safely, returning 'N/A' for missing values."""
        if val is None or val == "N/A":
            return "N/A"
        try:
            return f"{float(val):{fmt}}"
        except (ValueError, TypeError):
            return str(val)

    lines += [
        "",
        "## Home Hospital (Hospital 0) — Test Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Test Loss | {_fmt(home.get('test_loss'))} |",
        f"| Accuracy | {_fmt(home.get('accuracy'))} |",
        f"| Precision (macro) | {_fmt(home.get('precision_macro'))} |",
        f"| Recall (macro) | {_fmt(home.get('recall_macro'))} |",
        f"| F1 (macro) | {_fmt(home.get('f1_macro'))} |",
        f"| ROC-AUC | {_fmt(home.get('roc_auc'))} |",
        f"| Samples | {home.get('num_samples', 'N/A')} |",
        "",
    ]

    if home.get("confusion_matrix"):
        lines += [
            "### Confusion Matrix",
            "",
            "```",
            str(home["confusion_matrix"]),
            "```",
            "",
        ]

    # Cross-hospital table
    if cross:
        lines += [
            "## Cross-Hospital Generalization",
            "",
            "The model trained **only** on Hospital 0's data is evaluated",
            "on the test sets of other hospitals. This demonstrates the",
            "**small-data weakness**: a model trained on a single hospital's",
            "limited, non-IID data fails to generalise to unseen hospitals.",
            "",
            "| Hospital | Accuracy | Precision | Recall | F1 | ROC-AUC | Samples |",
            "|----------|----------|-----------|--------|----|---------|---------|",
        ]
        for hosp_name, hosp_metrics in cross.items():
            acc = hosp_metrics.get("accuracy", float("nan"))
            prec = hosp_metrics.get("precision_macro", float("nan"))
            rec = hosp_metrics.get("recall_macro", float("nan"))
            f1 = hosp_metrics.get("f1_macro", float("nan"))
            auc = hosp_metrics.get("roc_auc", float("nan"))
            n = hosp_metrics.get("num_samples", "N/A")
            lines.append(
                f"| {hosp_name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {n} |"
            )

        lines += [
            "",
            "> **Interpretation**: The drop in accuracy and F1 on other hospitals'",
            "> data compared to Hospital 0 confirms the small-data weakness.",
            "> Federated learning aims to close this gap by jointly training",
            "> across all hospitals without sharing raw data.",
            "",
        ]

    lines += [
        "## Reproducibility",
        "",
        f"- **Seed**: `{seed_val}`",
        "- **Config file**: `data/checkpoints/baseline/baseline_config.json`",
        "- **Checkpoint**: `data/checkpoints/baseline/baseline_best_model.pt`",
        "",
        "To reproduce, run:",
        "",
        "```bash",
        "cd shared && python -m medshield.model.run_baseline",
        "```",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote Markdown report to %s", md_path)

    return md_path, json_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_baseline(
    *,
    num_patients: int = 30,
    slices_per_patient: int = 5,
    num_hospitals: int = 3,
    epochs: int = 3,
    batch_size: int = 8,
    seed: int = SEED,
    synthetic_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    results_dir: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full single-hospital baseline workflow.

    Parameters
    ----------
    num_patients : int
        Number of synthetic patients to generate.
    slices_per_patient : int
        Slices per patient.
    num_hospitals : int
        Number of simulated hospitals for partitioning.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for dataloaders.
    seed : int
        Master random seed.
    synthetic_dir : Path | None
        Override for synthetic data directory.
    checkpoint_dir : Path | None
        Override for checkpoint directory.
    results_dir : Path | None
        Override for results directory.
    log_dir : Path | None
        Override for experiment log directory.

    Returns
    -------
    dict[str, Any]
        Full results dictionary.
    """
    synthetic_dir = synthetic_dir or DEFAULT_SYNTHETIC_DIR
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    log_dir = log_dir or DEFAULT_LOG_DIR

    # 1. Seed everything
    seed_everything(seed)

    # 2. Generate synthetic data
    logger.info("Step 1/8: Generating synthetic data...")
    manifest_path = generate_synthetic_data(
        data_dir=synthetic_dir,
        num_patients=num_patients,
        slices_per_patient=slices_per_patient,
        seed=seed,
    )

    # 3. Partition into hospitals
    logger.info("Step 2/8: Partitioning into %d hospitals...", num_hospitals)
    client_dfs = partition_by_patient(
        manifest_path=manifest_path,
        num_clients=num_hospitals,
        alpha=0.5,
        seed=seed,
    )

    # 4. Split Hospital 0 into train/val/test
    logger.info("Step 3/8: Creating train/val/test splits for Hospital 0...")
    train_df, val_df, test_df = create_hospital_splits(client_dfs[0], seed=seed)

    logger.info(
        "Hospital 0 splits — Train: %d, Val: %d, Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Get class counts for the training set
    class_counts = confirm_mapping(train_df["has_tumor"].tolist())

    # 5. Build DataLoaders
    logger.info("Step 4/8: Building DataLoaders...")
    train_loader = create_dataloaders(
        manifest_df=train_df,
        data_dir=synthetic_dir,
        batch_size=batch_size,
        is_train=True,
        num_workers=0,
        out_channels=3,
    )
    val_loader = create_dataloaders(
        manifest_df=val_df,
        data_dir=synthetic_dir,
        batch_size=batch_size,
        is_train=False,
        num_workers=0,
        out_channels=3,
    )
    test_loader = create_dataloaders(
        manifest_df=test_df,
        data_dir=synthetic_dir,
        batch_size=batch_size,
        is_train=False,
        num_workers=0,
        out_channels=3,
    )

    # 6. Create model, optimizer, scheduler, loss
    logger.info("Step 5/8: Creating model and training components...")
    config = ModelConfig(
        epochs=epochs,
        learning_rate=1e-4,
        weight_decay=1e-4,
        drop_rate=0.1,
        pretrained=True,
        num_classes=2,
    )

    device = get_device()
    model = create_model(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    loss_fn = get_loss_function(class_counts=class_counts, device=device)

    # 7. Set up experiment logger
    exp_logger = ExperimentLogger(
        log_dir=str(log_dir),
        run_id=f"baseline_seed{seed}",
    )
    exp_logger.log_hyperparameters(
        config,
        extra={
            "seed": seed,
            "num_patients": num_patients,
            "num_hospitals": num_hospitals,
            "hospital_trained_on": 0,
            "batch_size": batch_size,
        },
    )

    # 8. Train
    logger.info("Step 6/8: Training on Hospital 0...")
    start_time = time.time()
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=0,
        experiment_logger=exp_logger,
    )
    training_duration = time.time() - start_time
    logger.info("Training completed in %.1f seconds.", training_duration)

    # 9. Evaluate on Hospital 0 test set
    logger.info("Step 7/8: Evaluating on Hospital 0 test set...")
    home_metrics = evaluate_model(model, test_loader, loss_fn, device)
    logger.info("Hospital 0 test accuracy: %.4f", home_metrics.get("accuracy", float("nan")))

    # 10. Cross-hospital evaluation
    logger.info("Step 8/8: Cross-hospital evaluation...")
    cross_hospital_results: dict[str, dict[str, Any]] = {}

    for hosp_idx in range(num_hospitals):
        if hosp_idx == 0:
            continue  # Skip the home hospital

        hosp_name = f"Hospital {hosp_idx}"

        # Split this hospital's data
        _, _, other_test_df = create_hospital_splits(client_dfs[hosp_idx], seed=seed)

        if len(other_test_df) == 0:
            logger.warning("Hospital %d has no test data, skipping.", hosp_idx)
            continue

        other_test_loader = create_dataloaders(
            manifest_df=other_test_df,
            data_dir=synthetic_dir,
            batch_size=batch_size,
            is_train=False,
            num_workers=0,
            out_channels=3,
        )

        hosp_metrics = evaluate_model(model, other_test_loader, loss_fn, device)
        cross_hospital_results[hosp_name] = hosp_metrics

        logger.info(
            "%s test accuracy: %.4f",
            hosp_name,
            hosp_metrics.get("accuracy", float("nan")),
        )

    # 11. Assemble results
    results: dict[str, Any] = {
        "task": "Task 31 — Single-Hospital Baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training_duration_seconds": round(training_duration, 2),
        "config": config.to_dict(),
        "home_hospital": home_metrics,
        "cross_hospital": cross_hospital_results,
    }

    # 12. Save config alongside checkpoint
    config_path = checkpoint_dir / "baseline_config.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    logger.info("Saved config to %s", config_path)

    # 13. Save the final model checkpoint explicitly
    final_ckpt_path = str(checkpoint_dir / "baseline_best_model.pt")
    save_checkpoint(
        model=model,
        epoch=epochs,
        val_loss=home_metrics.get("test_loss", float("nan")),
        config=config,
        filepath=final_ckpt_path,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # 14. Write reports
    md_path, json_path = generate_results_report(results, config, results_dir)

    logger.info("=" * 60)
    logger.info("BASELINE COMPLETE")
    logger.info("  Checkpoint : %s", final_ckpt_path)
    logger.info("  Config     : %s", config_path)
    logger.info("  Report (MD): %s", md_path)
    logger.info("  Report (JSON): %s", json_path)
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_baseline()
