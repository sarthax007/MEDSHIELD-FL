"""Tests for medshield.data.loader — Task 12 acceptance criteria."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medshield.data.loader import (
    MRIVolume,
    VolumeLoadError,
    load_dataset,
    load_patient,
    load_volume,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic NIfTI files
# ---------------------------------------------------------------------------

_TEST_SHAPE = (16, 16, 8)


def _make_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    """Write *data* as a NIfTI file at *path*."""
    if affine is None:
        affine = np.eye(4)
        # Add some voxel spacing to the affine matrix (e.g. 1.5, 1.5, 2.0)
        affine[0, 0] = 1.5
        affine[1, 1] = 1.5
        affine[2, 2] = 2.0

    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _create_patient(
    root: Path,
    patient_id: str,
    *,
    shape: tuple[int, ...] = _TEST_SHAPE,
    skip_modalities: list[str] | None = None,
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
        _make_nifti(pdir / f"{patient_id}_{mod}.nii.gz", data)

    if "seg" not in skip:
        seg = rng.choice([0, 1, 2, 4], size=shape).astype(np.uint8)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_volume_valid(single_patient: Path) -> None:
    path = single_patient / "BraTS2021_00001_t1.nii.gz"
    vol = load_volume(path)

    # Basic dataclass attributes
    assert isinstance(vol, MRIVolume)
    assert vol.patient_id == "BraTS2021_00001"
    assert vol.modality == "t1"
    assert vol.source_path == path

    # Data properties
    assert vol.shape == _TEST_SHAPE
    assert vol.dtype == np.float32
    assert vol.data.shape == _TEST_SHAPE
    assert vol.data.dtype == np.float32

    # Spatial properties
    assert vol.affine.shape == (4, 4)
    # The helper sets affine diagonals to 1.5, 1.5, 2.0
    assert vol.voxel_spacing == (1.5, 1.5, 2.0)


def test_load_volume_custom_dtype(single_patient: Path) -> None:
    path = single_patient / "BraTS2021_00001_t1.nii.gz"
    vol = load_volume(path, dtype=np.float64)
    assert vol.dtype == np.float64


def test_load_volume_error_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.nii.gz"
    with pytest.raises(VolumeLoadError, match="file does not exist"):
        load_volume(path)


def test_load_volume_error_wrong_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(VolumeLoadError, match="unsupported extension"):
        load_volume(path)


def test_load_volume_error_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.nii.gz"
    path.write_text("this is not a real nifti file")
    with pytest.raises(VolumeLoadError, match="nibabel failed to load file"):
        load_volume(path)


def test_load_patient_valid(single_patient: Path) -> None:
    patient = load_patient(single_patient)

    assert isinstance(patient, dict)
    assert set(patient.keys()) == {"t1", "t1ce", "t2", "flair", "seg"}

    for mod, vol in patient.items():
        assert isinstance(vol, MRIVolume)
        assert vol.modality == mod
        assert vol.patient_id == single_patient.name
        assert vol.shape == _TEST_SHAPE


def test_load_patient_subset_modalities(single_patient: Path) -> None:
    patient = load_patient(single_patient, modalities=("t1", "t2"), include_seg=False)
    assert set(patient.keys()) == {"t1", "t2"}


def test_load_patient_missing_modality(single_patient: Path) -> None:
    # Delete the flair file
    (single_patient / "BraTS2021_00001_flair.nii.gz").unlink()

    with pytest.raises(VolumeLoadError, match="missing modality 'flair'"):
        load_patient(single_patient)


def test_load_dataset(dataset_dir: Path) -> None:
    generator = load_dataset(dataset_dir)

    patients_loaded = 0
    for patient in generator:
        patients_loaded += 1
        assert isinstance(patient, dict)
        assert "t1" in patient
        assert patient["t1"].shape == _TEST_SHAPE

    assert patients_loaded == 3


def test_load_dataset_empty(tmp_path: Path) -> None:
    with pytest.raises(VolumeLoadError, match="no patient directories found"):
        # Evaluating the generator forces execution
        list(load_dataset(tmp_path))
