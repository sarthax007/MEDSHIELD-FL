"""Data quality validation runner.

Task 20 — Produce a data-quality validation report.

This script scans partitioned datasets and splits to verify data integrity,
preventing corrupt images or patient overlap from reaching the training loop.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_hospital_splits(client_idx: int, splits_dir: Path, data_dir: Path) -> dict:
    """Validate the train/val/test splits for a single hospital.

    Returns
    -------
    dict
        Dictionary containing counts and anomaly flags.
    """
    report = {
        "client_idx": client_idx,
        "splits_found": [],
        "patient_overlap": False,
        "total_samples": 0,
        "corrupt_files": [],
        "inconsistent_shapes": [],
        "duplicates": [],
        "class_distribution": {},
        "expected_shape": None,
    }

    patient_sets = {}
    all_filenames = []
    all_raw_labels = []

    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"client_{client_idx}_{split}.csv"
        if not split_path.exists():
            continue

        report["splits_found"].append(split)
        df = pd.read_csv(split_path)
        report["total_samples"] += len(df)

        # Track patient overlap
        patient_sets[split] = set(df["patient_id"])

        # Track duplicates
        for fname in df["filename"]:
            if fname in all_filenames:
                report["duplicates"].append(fname)
            all_filenames.append(fname)

        # Track labels for class distribution
        all_raw_labels.extend(df["has_tumor"].tolist())

        # Validate physical files
        for _, row in df.iterrows():
            fpath = data_dir / row["filename"]
            if not fpath.exists():
                report["corrupt_files"].append(row["filename"])
                continue

            try:
                arr = np.load(fpath)
                shape = arr.shape
                if report["expected_shape"] is None:
                    report["expected_shape"] = shape
                elif shape != report["expected_shape"]:
                    report["inconsistent_shapes"].append((row["filename"], shape))
            except Exception:
                report["corrupt_files"].append(row["filename"])

    # Verify patient overlap
    train_p = patient_sets.get("train", set())
    val_p = patient_sets.get("val", set())
    test_p = patient_sets.get("test", set())

    if (
        (train_p and val_p and not train_p.isdisjoint(val_p))
        or (train_p and test_p and not train_p.isdisjoint(test_p))
        or (val_p and test_p and not val_p.isdisjoint(test_p))
    ):
        report["patient_overlap"] = True

    # Count class distributions
    if all_raw_labels:
        # We can just count directly, or use confirm_mapping if available
        tumors = sum(1 for x in all_raw_labels if x)
        no_tumors = len(all_raw_labels) - tumors
        report["class_distribution"] = {"Tumor": tumors, "No-Tumor": no_tumors}

    return report


def generate_report(data_dir: Path | str, splits_dir: Path | str, output_path: Path | str) -> None:
    """Run validation across all hospital clients and generate a Markdown report."""
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all clients by looking at client_*_train.csv files
    client_indices = set()
    for p in splits_dir.glob("client_*_train.csv"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            client_indices.add(int(parts[1]))

    client_indices = sorted(client_indices)

    if not client_indices:
        logger.warning(f"No client splits found in {splits_dir}")
        return

    md_lines = [
        "# Data Quality Validation Report",
        "",
        "This report was automatically generated to confirm data pipeline output is clean and ready for training.",
        "",
    ]

    global_samples = 0
    global_corrupt = 0
    global_overlap = False

    for c_idx in client_indices:
        logger.info(f"Validating Client {c_idx}...")
        rep = validate_hospital_splits(c_idx, splits_dir, data_dir)

        md_lines.extend(
            [
                f"## Hospital {c_idx}",
                f"- **Splits**: {', '.join(rep['splits_found'])}",
                f"- **Total Samples**: {rep['total_samples']}",
                f"- **Class Distribution**: {rep['class_distribution']}",
                f"- **Image Shape Consistency**: {'OK' if not rep['inconsistent_shapes'] else 'FAILED'}",
                f"- **Patient Overlap**: {'DETECTED' if rep['patient_overlap'] else 'None'}",
                f"- **Corrupt/Missing Files**: {len(rep['corrupt_files'])}",
                f"- **Duplicate Files**: {len(rep['duplicates'])}",
                "",
            ]
        )

        global_samples += rep["total_samples"]
        global_corrupt += len(rep["corrupt_files"])
        if rep["patient_overlap"]:
            global_overlap = True

    md_lines.extend(
        [
            "## Overall Summary",
            f"- **Total Hospitals**: {len(client_indices)}",
            f"- **Total Samples**: {global_samples}",
            f"- **Global Patient Leakage**: {'DETECTED' if global_overlap else 'None'}",
            f"- **Total Corrupt/Missing**: {global_corrupt}",
            "",
        ]
    )

    report_content = "\n".join(md_lines)
    output_path.write_text(report_content)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate data-quality validation report.")
    parser.add_argument("--data-dir", type=str, default="data/slices", help="Path to .npy slices")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to split CSVs")
    parser.add_argument(
        "--output", type=str, default="docs/data_quality_report.md", help="Output report path"
    )
    args = parser.parse_args()

    # Create dummy dirs if they don't exist just to prevent errors in empty projects
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)

    generate_report(args.data_dir, args.splits_dir, args.output)
