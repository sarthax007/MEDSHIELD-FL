"""Tests for medshield.data.slicer — Task 13 acceptance criteria."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from medshield.data.loader import MRIVolume
from medshield.data.slicer import (
    SliceExtractor,
    SliceStrategy,
    extract_dataset,
    select_central_slices,
    select_tumor_slices,
    select_uniform_slices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_SHAPE = (16, 16, 8)


def _make_dummy_volume(modality: str, patient_id: str) -> MRIVolume:
    """Create a dummy MRIVolume for testing without touching disk."""
    data = np.ones(_TEST_SHAPE, dtype=np.float32)
    # Give different modalities different values
    if modality == "t1ce":
        data *= 2.0
    elif modality == "t2":
        data *= 3.0
    elif modality == "flair":
        data *= 4.0
    elif modality == "seg":
        data = np.zeros(_TEST_SHAPE, dtype=np.uint8)
        # Put "tumor" pixels in slices 2, 4, 6
        data[5:10, 5:10, 2] = 1
        data[5:10, 5:10, 4] = 2
        data[5:10, 5:10, 6] = 4

    return MRIVolume(
        data=data,
        affine=np.eye(4),
        voxel_spacing=(1.0, 1.0, 1.0),
        patient_id=patient_id,
        modality=modality,
        source_path=Path(f"dummy/{modality}.nii.gz"),
    )


@pytest.fixture()
def dummy_patient() -> dict[str, MRIVolume]:
    patient_id = "BraTS2021_00000"
    return {
        mod: _make_dummy_volume(mod, patient_id) for mod in ("t1", "t1ce", "t2", "flair", "seg")
    }


# ---------------------------------------------------------------------------
# Tests — Slice Selection
# ---------------------------------------------------------------------------


def test_select_central_slices() -> None:
    # Depth 10, ask for 4 -> center is around 5
    # (10 - 4) // 2 = 3. Indices: 3, 4, 5, 6
    slices = select_central_slices(10, 4)
    assert slices == [3, 4, 5, 6]

    # Ask for more than depth -> returns all
    slices = select_central_slices(5, 10)
    assert slices == [0, 1, 2, 3, 4]


def test_select_uniform_slices() -> None:
    # Depth 9, ask for 3 -> should pick evenly
    slices = select_uniform_slices(9, 3)
    assert slices == [0, 4, 8]


def test_select_tumor_slices(dummy_patient: dict[str, MRIVolume]) -> None:
    seg = dummy_patient["seg"].data
    # Tumor is present in slices 2, 4, 6 with 25 pixels each (5x5 block)
    slices = select_tumor_slices(seg, min_tumor_pixels=10)
    assert slices == [2, 4, 6]

    # Require more pixels than present
    slices_empty = select_tumor_slices(seg, min_tumor_pixels=100)
    assert slices_empty == []


# ---------------------------------------------------------------------------
# Tests — Extractor
# ---------------------------------------------------------------------------


def test_slice_extractor_tumor_strategy(dummy_patient: dict[str, MRIVolume]) -> None:
    extractor = SliceExtractor(strategy=SliceStrategy.TUMOR_ONLY, min_tumor_pixels=10)
    extracted = extractor.extract(dummy_patient)

    assert len(extracted) == 3
    # Slices 2, 4, 6 have tumor
    assert [e.slice_idx for e in extracted] == [2, 4, 6]

    # Check data shape (4 modalities, 16x16)
    for ext in extracted:
        assert ext.data.shape == (4, 16, 16)
        assert ext.has_tumor is True
        assert ext.tumor_pixels == 25


def test_slice_extractor_channel_stacking(dummy_patient: dict[str, MRIVolume]) -> None:
    extractor = SliceExtractor(
        modalities=("t1", "t1ce", "t2", "flair"),
        strategy=SliceStrategy.ALL,
    )
    extracted = extractor.extract(dummy_patient)
    assert len(extracted) == 8  # all slices

    slice_0 = extracted[0]
    # Check that channels are stacked in the requested order
    # (dummy data sets T1=1, T1ce=2, T2=3, FLAIR=4)
    assert np.all(slice_0.data[0] == 1.0)
    assert np.all(slice_0.data[1] == 2.0)
    assert np.all(slice_0.data[2] == 3.0)
    assert np.all(slice_0.data[3] == 4.0)


def test_slice_extractor_missing_modality(dummy_patient: dict[str, MRIVolume]) -> None:
    del dummy_patient["t1ce"]
    extractor = SliceExtractor()
    with pytest.raises(ValueError, match="missing required modality"):
        extractor.extract(dummy_patient)


# ---------------------------------------------------------------------------
# Tests — Dataset Pipeline Integration
# ---------------------------------------------------------------------------


def test_extract_dataset_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # We mock load_dataset so we don't need real NIfTI files on disk for this test
    # but we can test the pipeline's file writing and manifest generation logic.
    def mock_load_dataset(*args, **kwargs):
        # Yield 2 patients
        yield {
            mod: _make_dummy_volume(mod, "BraTS2021_00000")
            for mod in ("t1", "t1ce", "t2", "flair", "seg")
        }
        yield {
            mod: _make_dummy_volume(mod, "BraTS2021_00001")
            for mod in ("t1", "t1ce", "t2", "flair", "seg")
        }

    import medshield.data.slicer as slicer_mod

    monkeypatch.setattr(slicer_mod, "load_dataset", mock_load_dataset)

    output_dir = tmp_path / "output"
    # Tumor only strategy -> 3 slices per patient -> 6 total slices
    extract_dataset(Path("dummy_in"), output_dir, strategy=SliceStrategy.TUMOR_ONLY)

    # Check that directory structure exists
    assert (output_dir / "npy").exists()

    # Check that 6 numpy files were written
    npy_files = list((output_dir / "npy").glob("*.npy"))
    assert len(npy_files) == 6

    # Load one to verify it's a valid numpy array
    loaded_data = np.load(npy_files[0])
    assert loaded_data.shape == (4, 16, 16)

    # Check manifest.csv
    manifest_path = output_dir / "manifest.csv"
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["filename", "patient_id", "slice_idx", "has_tumor", "tumor_pixels"]
        rows = list(reader)
        assert len(rows) == 6
        # Check first row logic
        # filename, patient_id, slice_idx, has_tumor, tumor_pixels
        assert rows[0][1].startswith("BraTS2021_0000")
        assert rows[0][3] == "True"
        assert rows[0][4] == "25"

    # Check that spot checks were generated
    spot_check_dir = output_dir / "spot_checks"
    assert spot_check_dir.exists()
    png_files = list(spot_check_dir.glob("*.png"))
    # The pipeline caps spot checks at 10 (or all if < 10)
    assert len(png_files) == 6
