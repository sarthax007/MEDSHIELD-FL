"""BraTS dataset inspection — sample slices, anomaly detection, and class distribution.

Task 11 — Inspect and document the BraTS structure.

Produces:
* Sample slice images for each MRI modality (saved to ``docs/results/``).
* An anomaly report listing missing modalities, inconsistent shapes, and
  unexpected label values.
* A tumor-class distribution summary (table + chart).

Usage::

    python -m medshield.data.inspect --data-dir data/raw
    python -m medshield.data.inspect --data-dir data/raw --output-dir docs/results
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITIES = ("t1", "t1ce", "t2", "flair")
SEG_SUFFIX = "seg"
ALL_SUFFIXES = (*MODALITIES, SEG_SUFFIX)

# BraTS 2021 valid segmentation labels (note: 3 is intentionally absent)
VALID_SEG_LABELS = frozenset({0, 1, 2, 4})

# Expected volume shape for BraTS 2021
EXPECTED_SHAPE = (240, 240, 155)


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------


@dataclass
class PatientInfo:
    """Metadata extracted from a single patient directory."""

    patient_id: str
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    voxel_spacings: dict[str, tuple[float, ...]] = field(default_factory=dict)
    intensity_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    label_counts: dict[int, int] = field(default_factory=dict)
    missing_modalities: list[str] = field(default_factory=list)
    has_nan: list[str] = field(default_factory=list)
    unexpected_labels: list[int] = field(default_factory=list)
    shape_consistent: bool = True


@dataclass
class DatasetReport:
    """Aggregated inspection report for the entire dataset."""

    total_patients: int = 0
    patients: list[PatientInfo] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    global_label_counts: Counter = field(default_factory=Counter)
    shape_distribution: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Core inspection logic
# ---------------------------------------------------------------------------


def _find_patient_dirs(data_dir: Path) -> list[Path]:
    """Return sorted list of patient directories under *data_dir*."""
    # Handle both data/raw and data/raw/BraTS2021_Training_Data layouts
    candidates = sorted(
        [d for d in data_dir.rglob("BraTS2021_*") if d.is_dir()],
        key=lambda p: p.name,
    )
    # Filter to only leaf-level patient dirs (those containing .nii.gz files)
    patient_dirs = [d for d in candidates if list(d.glob("*.nii.gz"))]
    return patient_dirs


def _find_modality_file(patient_dir: Path, modality: str) -> Path | None:
    """Locate the NIfTI file for a given modality inside *patient_dir*."""
    pattern = f"*_{modality}.nii.gz"
    matches = list(patient_dir.glob(pattern))
    return matches[0] if matches else None


def inspect_patient(patient_dir: Path) -> PatientInfo:
    """Inspect a single patient directory and return structured metadata."""
    info = PatientInfo(patient_id=patient_dir.name)

    for suffix in ALL_SUFFIXES:
        fpath = _find_modality_file(patient_dir, suffix)
        if fpath is None:
            info.missing_modalities.append(suffix)
            continue

        try:
            img = cast(nib.Nifti1Image, nib.load(str(fpath)))
            data = img.get_fdata()
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fpath, exc)
            info.missing_modalities.append(suffix)
            continue

        shape = tuple(data.shape)
        info.shapes[suffix] = shape

        # Record voxel spacing from the affine
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        info.voxel_spacings[suffix] = spacing

        if suffix == SEG_SUFFIX:
            # Label analysis
            unique_labels = set(np.unique(data).astype(int))
            info.label_counts = dict(Counter(data.flatten().astype(int)))
            unexpected = unique_labels - VALID_SEG_LABELS
            if unexpected:
                info.unexpected_labels = sorted(unexpected)
        else:
            # Intensity range
            info.intensity_ranges[suffix] = (float(np.min(data)), float(np.max(data)))
            # NaN / Inf check
            if not np.all(np.isfinite(data)):
                info.has_nan.append(suffix)

    # Check shape consistency across modalities
    shapes = list(info.shapes.values())
    if shapes and len(set(shapes)) > 1:
        info.shape_consistent = False

    return info


def inspect_dataset(data_dir: Path) -> DatasetReport:
    """Walk the entire dataset and produce an aggregated report."""
    report = DatasetReport()

    patient_dirs = _find_patient_dirs(data_dir)
    report.total_patients = len(patient_dirs)

    if report.total_patients == 0:
        report.anomalies.append(
            f"No patient directories found under {data_dir}. "
            "Ensure the BraTS dataset is downloaded and extracted."
        )
        return report

    logger.info("Inspecting %d patients …", report.total_patients)

    for pdir in patient_dirs:
        info = inspect_patient(pdir)
        report.patients.append(info)

        # Accumulate global label counts
        for label, count in info.label_counts.items():
            report.global_label_counts[label] += count

        # Track shape distribution
        for _suffix, shape in info.shapes.items():
            report.shape_distribution[shape] += 1

        # Record anomalies
        if info.missing_modalities:
            report.anomalies.append(
                f"{info.patient_id}: missing modalities {info.missing_modalities}"
            )
        if not info.shape_consistent:
            report.anomalies.append(
                f"{info.patient_id}: inconsistent shapes across modalities: {info.shapes}"
            )
        if info.unexpected_labels:
            report.anomalies.append(
                f"{info.patient_id}: unexpected segmentation labels {info.unexpected_labels}"
            )
        if info.has_nan:
            report.anomalies.append(f"{info.patient_id}: non-finite values in {info.has_nan}")
        # Check for zero-volume segmentation
        if SEG_SUFFIX not in info.missing_modalities:
            tumor_voxels = sum(v for k, v in info.label_counts.items() if k != 0)
            if tumor_voxels == 0:
                report.anomalies.append(f"{info.patient_id}: zero tumor voxels (all background)")

    return report


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def save_sample_slices(
    patient_dir: Path,
    output_dir: Path,
    *,
    slice_idx: int | None = None,
) -> list[Path]:
    """Save a central axial slice from each modality as a PNG image.

    Returns the list of saved file paths.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for suffix in (*MODALITIES, SEG_SUFFIX):
        fpath = _find_modality_file(patient_dir, suffix)
        if fpath is None:
            continue

        img = cast(nib.Nifti1Image, nib.load(str(fpath)))
        data = img.get_fdata()

        # Pick the central axial slice if none specified
        idx = slice_idx if slice_idx is not None else data.shape[2] // 2

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cmap = "nipy_spectral" if suffix == SEG_SUFFIX else "gray"
        ax.imshow(data[:, :, idx].T, cmap=cmap, origin="lower")
        ax.set_title(f"{patient_dir.name} — {suffix.upper()} (slice {idx})")
        ax.axis("off")

        out_path = output_dir / f"{patient_dir.name}_{suffix}_slice{idx}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved.append(out_path)
        logger.info("Saved %s", out_path)

    return saved


def save_class_distribution_chart(
    report: DatasetReport,
    output_dir: Path,
) -> Path:
    """Save a bar chart of the tumor label distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing"}
    labels_sorted = [0, 1, 2, 4]
    counts = [report.global_label_counts.get(lbl, 0) for lbl in labels_sorted]
    names = [label_names.get(lbl, str(lbl)) for lbl in labels_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart — all classes
    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981"]
    axes[0].bar(names, counts, color=colors)
    axes[0].set_title("Voxel counts per label class")
    axes[0].set_ylabel("Voxel count")
    axes[0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Pie chart — tumor classes only (exclude background)
    tumor_counts = [c for lbl, c in zip(labels_sorted, counts, strict=False) if lbl != 0]
    tumor_names = [n for lbl, n in zip(labels_sorted, names, strict=False) if lbl != 0]
    tumor_colors = colors[1:]
    if any(c > 0 for c in tumor_counts):
        axes[1].pie(
            tumor_counts,
            labels=tumor_names,
            colors=tumor_colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("Tumor sub-class distribution (excluding background)")
    else:
        axes[1].text(0.5, 0.5, "No tumor data available", ha="center", va="center")
        axes[1].set_title("Tumor sub-class distribution")

    fig.tight_layout()
    out_path = output_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved class distribution chart → %s", out_path)
    return out_path


def save_anomaly_report(report: DatasetReport, output_dir: Path) -> Path:
    """Write the anomaly list to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "anomalies.json"

    payload = {
        "total_patients": report.total_patients,
        "anomaly_count": len(report.anomalies),
        "anomalies": report.anomalies,
        "shape_distribution": {str(k): v for k, v in report.shape_distribution.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Anomaly report → %s (%d issues)", out_path, len(report.anomalies))
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the inspection pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the BraTS 2021 dataset and generate a report."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing the BraTS patient folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results"),
        help="Directory to write slice images, charts, and anomaly report.",
    )
    parser.add_argument(
        "--sample-patients",
        type=int,
        default=3,
        help="Number of patients to generate sample slices for (default: 3).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    # --- Full dataset inspection -----------------------------------------------
    report = inspect_dataset(data_dir)

    if report.total_patients == 0:
        logger.error("No patients found — aborting.")
        raise SystemExit(1)

    # --- Sample slices ---------------------------------------------------------
    patient_dirs = _find_patient_dirs(data_dir)
    sample_count = min(args.sample_patients, len(patient_dirs))
    for pdir in patient_dirs[:sample_count]:
        save_sample_slices(pdir, output_dir / "sample_slices")

    # --- Class distribution chart ----------------------------------------------
    save_class_distribution_chart(report, output_dir)

    # --- Anomaly report --------------------------------------------------------
    save_anomaly_report(report, output_dir)

    # --- Summary ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Inspection complete — %d patients", report.total_patients)
    logger.info("Anomalies: %d", len(report.anomalies))
    logger.info("Shape distribution: %s", dict(report.shape_distribution))
    logger.info(
        "Label distribution: %s",
        {k: report.global_label_counts[k] for k in sorted(report.global_label_counts)},
    )
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
