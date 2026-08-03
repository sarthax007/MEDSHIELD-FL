"""MRI volume loader — reads NIfTI volumes into arrays with orientation and spacing.

Task 12 — Implement the MRI volume loader.

Provides a clean API for loading BraTS 2021 NIfTI files:

* :func:`load_volume` loads a single ``.nii.gz`` file → :class:`MRIVolume`.
* :func:`load_patient` loads all modalities for one patient → ``dict[str, MRIVolume]``.
* :func:`load_dataset` iterates over every patient in a dataset directory.

All functions raise :class:`VolumeLoadError` on failure with a descriptive message.

Usage::

    from medshield.data.loader import load_volume, load_patient

    vol = load_volume("data/raw/BraTS2021_00000/BraTS2021_00000_t1.nii.gz")
    print(vol.shape, vol.voxel_spacing)

    patient = load_patient("data/raw/BraTS2021_00000")
    for modality, vol in patient.items():
        print(modality, vol.shape)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
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

_VALID_EXTENSIONS = frozenset({".nii", ".nii.gz"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VolumeLoadError(Exception):
    """Raised when an MRI volume cannot be loaded or is invalid.

    Attributes
    ----------
    path : Path | None
        The file path that triggered the error, if available.
    reason : str
        A human-readable description of what went wrong.
    """

    def __init__(self, reason: str, *, path: Path | None = None) -> None:
        self.path = path
        self.reason = reason
        if path is not None:
            super().__init__(f"{path}: {reason}")
        else:
            super().__init__(reason)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MRIVolume:
    """A loaded 3-D MRI volume with associated spatial metadata.

    Parameters
    ----------
    data : np.ndarray
        The voxel data, shape ``(H, W, D)`` with dtype ``float32``.
    affine : np.ndarray
        The 4×4 affine matrix mapping voxel indices to world coordinates.
    voxel_spacing : tuple[float, float, float]
        Voxel size in mm along each spatial axis (extracted from the affine).
    patient_id : str
        Identifier for the patient (typically the directory name).
    modality : str
        The MRI modality or ``"seg"`` for the segmentation mask.
    source_path : Path
        Absolute path to the NIfTI file this volume was loaded from.
    """

    data: np.ndarray
    affine: np.ndarray
    voxel_spacing: tuple[float, float, float]
    patient_id: str
    modality: str
    source_path: Path

    # Convenience properties ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial dimensions of the volume ``(H, W, D)``."""
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype of the voxel data."""
        return self.data.dtype


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_valid_extension(path: Path) -> bool:
    """Check whether *path* has a recognised NIfTI extension."""
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _find_modality_file(patient_dir: Path, modality: str) -> Path | None:
    """Locate the NIfTI file for *modality* inside *patient_dir*."""
    pattern = f"*_{modality}.nii.gz"
    matches = list(patient_dir.glob(pattern))
    if not matches:
        # Fall back to uncompressed NIfTI
        pattern_plain = f"*_{modality}.nii"
        matches = list(patient_dir.glob(pattern_plain))
    return matches[0] if matches else None


def _find_patient_dirs(data_dir: Path) -> list[Path]:
    """Return sorted list of patient directories under *data_dir*."""
    candidates = sorted(
        [d for d in data_dir.rglob("BraTS2021_*") if d.is_dir()],
        key=lambda p: p.name,
    )
    return [d for d in candidates if list(d.glob("*.nii*"))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_volume(
    path: str | Path,
    *,
    dtype: np.dtype | type = np.float32,
    patient_id: str = "",
    modality: str = "",
) -> MRIVolume:
    """Load a single NIfTI volume from *path*.

    Parameters
    ----------
    path : str | Path
        File path to a ``.nii`` or ``.nii.gz`` file.
    dtype : np.dtype | type
        Target dtype for the returned array (default: ``float32``).
    patient_id : str
        Optional patient identifier; inferred from the parent directory name
        if not supplied.
    modality : str
        Optional modality name; inferred from the filename if not supplied.

    Returns
    -------
    MRIVolume
        The loaded volume with spatial metadata.

    Raises
    ------
    VolumeLoadError
        If the file does not exist, has an invalid extension, or cannot be
        parsed by nibabel.
    """
    filepath = Path(path).resolve()

    # --- Validate file existence ---------------------------------------------
    if not filepath.exists():
        raise VolumeLoadError("file does not exist", path=filepath)

    # --- Validate extension ---------------------------------------------------
    if not _has_valid_extension(filepath):
        raise VolumeLoadError(
            f"unsupported extension (expected .nii or .nii.gz, got " f"'{filepath.suffix}')",
            path=filepath,
        )

    # --- Load via nibabel -----------------------------------------------------
    try:
        img = cast(nib.Nifti1Image, nib.load(str(filepath)))
    except Exception as exc:
        raise VolumeLoadError(
            f"nibabel failed to load file: {exc}",
            path=filepath,
        ) from exc

    try:
        data = np.asarray(img.get_fdata(), dtype=dtype)
    except Exception as exc:
        raise VolumeLoadError(
            f"failed to extract voxel data: {exc}",
            path=filepath,
        ) from exc

    # --- Validate dimensionality ----------------------------------------------
    if data.ndim != 3:
        raise VolumeLoadError(
            f"expected a 3-D volume, got {data.ndim}-D array with shape {data.shape}",
            path=filepath,
        )

    # --- Extract spatial metadata --------------------------------------------
    affine = np.array(img.affine, dtype=np.float64)
    zooms = img.header.get_zooms()[:3]
    voxel_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    # --- Infer patient_id / modality if not provided -------------------------
    if not patient_id:
        patient_id = filepath.parent.name

    if not modality:
        stem = filepath.name.replace(".nii.gz", "").replace(".nii", "")
        parts = stem.rsplit("_", 1)
        modality = parts[-1] if len(parts) > 1 else stem

    logger.debug(
        "Loaded %s/%s: shape=%s dtype=%s spacing=%s",
        patient_id,
        modality,
        data.shape,
        data.dtype,
        voxel_spacing,
    )

    return MRIVolume(
        data=data,
        affine=affine,
        voxel_spacing=voxel_spacing,
        patient_id=patient_id,
        modality=modality,
        source_path=filepath,
    )


def load_patient(
    patient_dir: str | Path,
    *,
    modalities: tuple[str, ...] | None = None,
    include_seg: bool = True,
    dtype: np.dtype | type = np.float32,
) -> dict[str, MRIVolume]:
    """Load all modalities for a single patient.

    Parameters
    ----------
    patient_dir : str | Path
        Path to the patient directory (e.g. ``data/raw/BraTS2021_00000``).
    modalities : tuple[str, ...] | None
        Which MRI modalities to load. Defaults to all four
        (``t1``, ``t1ce``, ``t2``, ``flair``).
    include_seg : bool
        Whether to also load the segmentation mask (default ``True``).
    dtype : np.dtype | type
        Target dtype for the voxel arrays.

    Returns
    -------
    dict[str, MRIVolume]
        Mapping from modality name (or ``"seg"``) to :class:`MRIVolume`.

    Raises
    ------
    VolumeLoadError
        If the directory does not exist or an expected modality file is
        missing.
    """
    pdir = Path(patient_dir).resolve()

    if not pdir.is_dir():
        raise VolumeLoadError(
            f"patient directory does not exist: {pdir}",
        )

    patient_id = pdir.name
    suffixes_to_load = list(modalities or MODALITIES)
    if include_seg:
        suffixes_to_load.append(SEG_SUFFIX)

    volumes: dict[str, MRIVolume] = {}

    for suffix in suffixes_to_load:
        fpath = _find_modality_file(pdir, suffix)
        if fpath is None:
            raise VolumeLoadError(
                f"missing modality '{suffix}' — no file matching "
                f"'*_{suffix}.nii.gz' found in {pdir}",
                path=pdir,
            )
        volumes[suffix] = load_volume(
            fpath,
            dtype=dtype,
            patient_id=patient_id,
            modality=suffix,
        )

    logger.info(
        "Loaded patient %s: %d volumes",
        patient_id,
        len(volumes),
    )
    return volumes


def load_dataset(
    data_dir: str | Path,
    *,
    modalities: tuple[str, ...] | None = None,
    include_seg: bool = True,
    dtype: np.dtype | type = np.float32,
) -> Generator[dict[str, MRIVolume], None, None]:
    """Iterate over patients in the dataset, yielding one patient at a time.

    This is a memory-efficient generator — only one patient's volumes are
    loaded into memory at any given time.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing patient subdirectories.
    modalities : tuple[str, ...] | None
        Which modalities to load (default: all four).
    include_seg : bool
        Whether to include segmentation masks.
    dtype : np.dtype | type
        Target dtype for voxel arrays.

    Yields
    ------
    dict[str, MRIVolume]
        Per-patient mapping from modality name to loaded volume.

    Raises
    ------
    VolumeLoadError
        If no patient directories are found.
    """
    root = Path(data_dir).resolve()
    patient_dirs = _find_patient_dirs(root)

    if not patient_dirs:
        raise VolumeLoadError(
            f"no patient directories found under {root}",
        )

    logger.info("Loading dataset: %d patients found", len(patient_dirs))

    for pdir in patient_dirs:
        yield load_patient(
            pdir,
            modalities=modalities,
            include_seg=include_seg,
            dtype=dtype,
        )
