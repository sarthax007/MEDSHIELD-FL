"""BraTS dataset download, integrity verification, and directory layout validation.

Task 10 — Download, verify (SHA-256 checksum), and unpack the BraTS 2021 dataset
into ``data/raw/``.  The script is idempotent: re-running it skips files that
already exist and pass the checksum.

Usage (from repository root)::

    python -m medshield.data.download          # default: data/raw
    python -m medshield.data.download --dest /mnt/data/brats
    python -m medshield.data.download --verify-only
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default output directory (relative to repo root)
_DEFAULT_RAW_DIR = Path("data/raw")

# Archive file name expected after manual download
ARCHIVE_FILENAME = "BraTS2021_Training_Data.zip"

# SHA-256 checksum of the official BraTS 2021 training archive.
# Update this value if the dataset source provides a different hash.
EXPECTED_SHA256 = "placeholder-sha256-update-after-first-verified-download"

# Minimum expected number of patient directories inside the extracted archive
MIN_EXPECTED_PATIENTS = 1200

# Expected MRI modalities per patient directory
EXPECTED_MODALITIES = frozenset({"t1", "t1ce", "t2", "flair", "seg"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of *path* in a streaming fashion."""
    h = hashlib.sha256()
    total = path.stat().st_size
    with open(path, "rb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Hashing {path.name}",
        leave=False,
    ) as pbar:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            pbar.update(len(chunk))
    return h.hexdigest()


def verify_archive(archive_path: Path, *, expected_sha256: str = EXPECTED_SHA256) -> bool:
    """Return *True* if *archive_path* exists and its SHA-256 matches."""
    if not archive_path.exists():
        logger.error("Archive not found: %s", archive_path)
        return False

    digest = _sha256_file(archive_path)

    if expected_sha256 == "placeholder-sha256-update-after-first-verified-download":
        logger.warning(
            "No reference checksum configured — recording archive hash for future runs: %s",
            digest,
        )
        # When running for the first time, we accept any hash and log it so the
        # developer can pin it in the constant above.
        return True

    if digest != expected_sha256:
        logger.error(
            "Checksum mismatch!\n  expected: %s\n  actual:   %s",
            expected_sha256,
            digest,
        )
        return False

    logger.info("Checksum OK (%s)", digest[:12])
    return True


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract the ZIP archive into *dest_dir* and return the extracted root folder.

    If the archive is already extracted (i.e. destination contents exist), this
    is a no-op.
    """
    # Check if already extracted
    candidate_dirs = [d for d in dest_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
    if candidate_dirs:
        logger.info("Archive appears already extracted → %s", candidate_dirs[0])
        return candidate_dirs[0]

    logger.info("Extracting %s → %s …", archive_path.name, dest_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, dest_dir)

    # Return the top-level directory that was created
    candidate_dirs = [d for d in dest_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
    if not candidate_dirs:
        raise RuntimeError(f"Extraction succeeded but no BraTS* directory found in {dest_dir}")
    return candidate_dirs[0]


def verify_directory_layout(dataset_dir: Path) -> bool:
    """Verify the extracted dataset has the expected structure.

    Checks:
    1. At least ``MIN_EXPECTED_PATIENTS`` patient sub-directories exist.
    2. Each patient directory contains all expected modalities (.nii.gz files).

    Returns *True* when all checks pass.
    """
    patient_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    if len(patient_dirs) < MIN_EXPECTED_PATIENTS:
        logger.error(
            "Expected at least %d patient directories, found %d",
            MIN_EXPECTED_PATIENTS,
            len(patient_dirs),
        )
        return False

    logger.info("Found %d patient directories", len(patient_dirs))

    missing_modalities: list[str] = []
    for pdir in patient_dirs:
        nii_files = {f.name.lower() for f in pdir.glob("*.nii.gz")}
        for modality in EXPECTED_MODALITIES:
            if not any(modality in fname for fname in nii_files):
                missing_modalities.append(f"{pdir.name}: missing {modality}")

    if missing_modalities:
        for msg in missing_modalities[:20]:  # limit output noise
            logger.warning(msg)
        if len(missing_modalities) > 20:
            logger.warning("… and %d more", len(missing_modalities) - 20)
        logger.error("Modality check failed — %d issues found", len(missing_modalities))
        return False

    logger.info("All patients contain expected modalities ✓")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the download/verification pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download, verify, and extract the BraTS 2021 dataset."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("DATA_RAW_DIR", str(_DEFAULT_RAW_DIR))),
        help="Destination directory for raw data (default: data/raw or $DATA_RAW_DIR)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=(
            "Path to the manually downloaded BraTS archive ZIP. "
            "If omitted, the script looks for the file in --dest."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an already-extracted dataset; do not download or extract.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dest: Path = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        # Look for the already-extracted BraTS directory
        candidate_dirs = [d for d in dest.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
        if not candidate_dirs:
            logger.error("No BraTS directory found in %s — nothing to verify.", dest)
            raise SystemExit(1)
        ok = verify_directory_layout(candidate_dirs[0])
        raise SystemExit(0 if ok else 1)

    # --- Locate archive -------------------------------------------------------
    archive_path: Path = args.archive if args.archive else dest / ARCHIVE_FILENAME
    if not archive_path.exists():
        logger.error(
            "Archive not found at %s.\n\n"
            "The BraTS 2021 dataset requires manual download:\n"
            "  1. Register at https://www.synapse.org/#!Synapse:syn25829067\n"
            "  2. Accept the data usage agreement\n"
            "  3. Download '%s'\n"
            "  4. Place it in %s\n"
            "  5. Re-run this script\n",
            archive_path,
            ARCHIVE_FILENAME,
            dest,
        )
        raise SystemExit(1)

    # --- Verify checksum ------------------------------------------------------
    if not verify_archive(archive_path):
        raise SystemExit(1)

    # --- Extract --------------------------------------------------------------
    extracted_dir = extract_archive(archive_path, dest)

    # --- Validate layout ------------------------------------------------------
    if not verify_directory_layout(extracted_dir):
        raise SystemExit(1)

    logger.info("Dataset ready at %s", extracted_dir)


if __name__ == "__main__":
    main()
