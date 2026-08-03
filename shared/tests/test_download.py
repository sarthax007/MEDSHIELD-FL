"""Tests for medshield.data.download — Task 10 acceptance criteria.

These tests use synthetic fixtures (tiny files / directories) so they run
without the real BraTS archive.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from medshield.data.download import (
    EXPECTED_MODALITIES,
    _sha256_file,
    extract_archive,
    verify_archive,
    verify_directory_layout,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Create a small file with known content for checksum tests."""
    p = tmp_path / "sample.bin"
    p.write_bytes(b"MedShield-FL test payload")
    return p


@pytest.fixture()
def valid_dataset(tmp_path: Path) -> Path:
    """Create a mock BraTS directory with the expected modality files."""
    dataset_dir = tmp_path / "BraTS2021_Training_Data"
    dataset_dir.mkdir()

    modalities = list(EXPECTED_MODALITIES)
    # Create 1250 synthetic patient directories (> MIN_EXPECTED_PATIENTS)
    for i in range(1250):
        patient_dir = dataset_dir / f"BraTS2021_{i:05d}"
        patient_dir.mkdir()
        for mod in modalities:
            nii = patient_dir / f"BraTS2021_{i:05d}_{mod}.nii.gz"
            nii.write_bytes(b"\x00")  # placeholder
    return dataset_dir


@pytest.fixture()
def sparse_dataset(tmp_path: Path) -> Path:
    """Create a dataset with too few patient directories."""
    dataset_dir = tmp_path / "BraTS2021_Training_Data"
    dataset_dir.mkdir()
    for i in range(5):
        patient_dir = dataset_dir / f"BraTS2021_{i:05d}"
        patient_dir.mkdir()
        for mod in EXPECTED_MODALITIES:
            nii = patient_dir / f"BraTS2021_{i:05d}_{mod}.nii.gz"
            nii.write_bytes(b"\x00")
    return dataset_dir


# ---------------------------------------------------------------------------
# Tests — SHA-256 hashing
# ---------------------------------------------------------------------------


class TestSha256:
    def test_correct_hash(self, sample_file: Path) -> None:
        expected = hashlib.sha256(b"MedShield-FL test payload").hexdigest()
        assert _sha256_file(sample_file) == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert _sha256_file(a) != _sha256_file(b)


# ---------------------------------------------------------------------------
# Tests — archive verification
# ---------------------------------------------------------------------------


class TestVerifyArchive:
    def test_missing_archive_returns_false(self, tmp_path: Path) -> None:
        assert verify_archive(tmp_path / "nonexistent.zip") is False

    def test_matching_checksum_returns_true(self, sample_file: Path) -> None:
        digest = hashlib.sha256(b"MedShield-FL test payload").hexdigest()
        assert verify_archive(sample_file, expected_sha256=digest) is True

    def test_wrong_checksum_returns_false(self, sample_file: Path) -> None:
        assert verify_archive(sample_file, expected_sha256="bad" * 16) is False

    def test_placeholder_checksum_always_passes(self, sample_file: Path) -> None:
        """When the placeholder constant is used, any file passes."""
        placeholder = "placeholder-sha256-update-after-first-verified-download"
        assert verify_archive(sample_file, expected_sha256=placeholder) is True


# ---------------------------------------------------------------------------
# Tests — archive extraction
# ---------------------------------------------------------------------------


class TestExtractArchive:
    def test_extracts_zip(self, tmp_path: Path) -> None:
        # Create a zip with a BraTS-like directory inside
        archive = tmp_path / "archive.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("BraTS2021_Data/patient_001/scan.nii.gz", b"\x00")

        result = extract_archive(archive, extract_to)
        assert result.name.startswith("BraTS")
        assert (result / "patient_001" / "scan.nii.gz").exists()

    def test_idempotent_no_re_extract(self, tmp_path: Path) -> None:
        """If a BraTS directory already exists, extraction is skipped."""
        dest = tmp_path / "dest"
        dest.mkdir()
        existing = dest / "BraTS2021_Already"
        existing.mkdir()
        (existing / "marker.txt").write_text("exists")

        # Should return the existing directory without needing an archive
        result = extract_archive(Path("dummy.zip"), dest)
        assert result == existing


# ---------------------------------------------------------------------------
# Tests — directory layout verification
# ---------------------------------------------------------------------------


class TestVerifyDirectoryLayout:
    def test_valid_layout_passes(self, valid_dataset: Path) -> None:
        assert verify_directory_layout(valid_dataset) is True

    def test_too_few_patients_fails(self, sparse_dataset: Path) -> None:
        assert verify_directory_layout(sparse_dataset) is False

    def test_missing_modality_fails(self, valid_dataset: Path) -> None:
        """Remove one modality file from the first patient and expect failure."""
        first_patient = sorted(valid_dataset.iterdir())[0]
        # Delete the first .nii.gz file
        nii_files = list(first_patient.glob("*.nii.gz"))
        nii_files[0].unlink()
        assert verify_directory_layout(valid_dataset) is False
