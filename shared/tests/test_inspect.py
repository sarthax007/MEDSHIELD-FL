"""Tests for medshield.data.inspect — Task 11 acceptance criteria.

Uses synthetic NIfTI volumes so the real dataset is not required.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medshield.data.inspect import (
    VALID_SEG_LABELS,
    inspect_dataset,
    inspect_patient,
    save_anomaly_report,
    save_class_distribution_chart,
    save_sample_slices,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic NIfTI files
# ---------------------------------------------------------------------------

# Use smaller volumes for fast tests
_TEST_SHAPE = (16, 16, 8)


def _make_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    """Write *data* as a NIfTI file at *path*."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _create_patient(
    root: Path,
    patient_id: str,
    *,
    shape: tuple[int, ...] = _TEST_SHAPE,
    skip_modalities: list[str] | None = None,
    bad_labels: list[int] | None = None,
    inject_nan: bool = False,
) -> Path:
    """Create a synthetic patient directory with NIfTI volumes."""
    pdir = root / patient_id
    pdir.mkdir(parents=True, exist_ok=True)

    skip = set(skip_modalities or [])
    rng = np.random.default_rng(42)

    for mod in ("t1", "t1ce", "t2", "flair"):
        if mod in skip:
            continue
        data = rng.random(shape, dtype=np.float32) * 3000.0
        if inject_nan and mod == "t1":
            data[0, 0, 0] = np.nan
        _make_nifti(pdir / f"{patient_id}_{mod}.nii.gz", data)

    if "seg" not in skip:
        labels = [0, 1, 2, 4]
        if bad_labels:
            labels.extend(bad_labels)
        seg = rng.choice(labels, size=shape).astype(np.uint8)
        _make_nifti(pdir / f"{patient_id}_seg.nii.gz", seg)

    return pdir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_patient(tmp_path: Path) -> Path:
    """One valid patient directory."""
    return _create_patient(tmp_path, "BraTS2021_00001")


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    """A small dataset with 3 valid patients."""
    root = tmp_path / "BraTS2021_Training_Data"
    root.mkdir()
    for i in range(3):
        _create_patient(root, f"BraTS2021_{i:05d}")
    return root


@pytest.fixture()
def dataset_with_anomalies(tmp_path: Path) -> Path:
    """A dataset containing various anomalies."""
    root = tmp_path / "BraTS2021_Training_Data"
    root.mkdir()
    # Normal patient
    _create_patient(root, "BraTS2021_00000")
    # Missing a modality
    _create_patient(root, "BraTS2021_00001", skip_modalities=["flair"])
    # Unexpected labels
    _create_patient(root, "BraTS2021_00002", bad_labels=[3, 5])
    # NaN in a modality
    _create_patient(root, "BraTS2021_00003", inject_nan=True)
    return root


# ---------------------------------------------------------------------------
# Tests — single patient inspection
# ---------------------------------------------------------------------------


class TestInspectPatient:
    def test_valid_patient_no_anomalies(self, single_patient: Path) -> None:
        info = inspect_patient(single_patient)
        assert info.patient_id == "BraTS2021_00001"
        assert len(info.missing_modalities) == 0
        assert info.shape_consistent is True
        assert len(info.unexpected_labels) == 0
        assert len(info.has_nan) == 0

    def test_records_shapes(self, single_patient: Path) -> None:
        info = inspect_patient(single_patient)
        assert "t1" in info.shapes
        assert "seg" in info.shapes
        assert info.shapes["t1"] == _TEST_SHAPE

    def test_records_intensity_ranges(self, single_patient: Path) -> None:
        info = inspect_patient(single_patient)
        for mod in ("t1", "t1ce", "t2", "flair"):
            lo, hi = info.intensity_ranges[mod]
            assert lo < hi

    def test_records_label_counts(self, single_patient: Path) -> None:
        info = inspect_patient(single_patient)
        assert len(info.label_counts) > 0
        # All labels should be valid
        for label in info.label_counts:
            assert label in VALID_SEG_LABELS

    def test_detects_missing_modality(self, tmp_path: Path) -> None:
        pdir = _create_patient(tmp_path, "BraTS2021_99999", skip_modalities=["t2"])
        info = inspect_patient(pdir)
        assert "t2" in info.missing_modalities

    def test_detects_unexpected_labels(self, tmp_path: Path) -> None:
        pdir = _create_patient(tmp_path, "BraTS2021_99998", bad_labels=[3, 7])
        info = inspect_patient(pdir)
        assert len(info.unexpected_labels) > 0

    def test_detects_nan(self, tmp_path: Path) -> None:
        pdir = _create_patient(tmp_path, "BraTS2021_99997", inject_nan=True)
        info = inspect_patient(pdir)
        assert "t1" in info.has_nan


# ---------------------------------------------------------------------------
# Tests — full dataset inspection
# ---------------------------------------------------------------------------


class TestInspectDataset:
    def test_counts_patients(self, dataset_dir: Path) -> None:
        report = inspect_dataset(dataset_dir)
        assert report.total_patients == 3

    def test_no_anomalies_for_clean_data(self, dataset_dir: Path) -> None:
        report = inspect_dataset(dataset_dir)
        assert len(report.anomalies) == 0

    def test_detects_anomalies(self, dataset_with_anomalies: Path) -> None:
        report = inspect_dataset(dataset_with_anomalies)
        assert len(report.anomalies) > 0
        anomaly_text = "\n".join(report.anomalies)
        assert "missing modalities" in anomaly_text
        assert "unexpected segmentation labels" in anomaly_text
        assert "non-finite values" in anomaly_text

    def test_accumulates_global_label_counts(self, dataset_dir: Path) -> None:
        report = inspect_dataset(dataset_dir)
        assert 0 in report.global_label_counts  # background always present

    def test_empty_dir_reports_zero_patients(self, tmp_path: Path) -> None:
        report = inspect_dataset(tmp_path)
        assert report.total_patients == 0
        assert len(report.anomalies) == 1  # "no patient dirs found" warning


# ---------------------------------------------------------------------------
# Tests — visualisation outputs
# ---------------------------------------------------------------------------


class TestVisualisations:
    def test_sample_slices_saved(self, single_patient: Path, tmp_path: Path) -> None:
        output = tmp_path / "slices"
        saved = save_sample_slices(single_patient, output)
        assert len(saved) == 5  # 4 modalities + seg
        for f in saved:
            assert f.exists()
            assert f.suffix == ".png"

    def test_class_distribution_chart(self, dataset_dir: Path, tmp_path: Path) -> None:
        report = inspect_dataset(dataset_dir)
        out = tmp_path / "charts"
        path = save_class_distribution_chart(report, out)
        assert path.exists()
        assert path.name == "class_distribution.png"

    def test_anomaly_report_json(self, dataset_with_anomalies: Path, tmp_path: Path) -> None:
        report = inspect_dataset(dataset_with_anomalies)
        out = tmp_path / "reports"
        path = save_anomaly_report(report, out)
        assert path.exists()
        assert path.name == "anomalies.json"

        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["anomaly_count"] > 0
        assert isinstance(data["anomalies"], list)
