"""Extract 2D slices from 3D MRI volumes.

Task 13 — Extract 2D slices for classification.

Converts 3D BraTS volumes into representative 2D multi-channel slices.
Extracts a configured subset of slices along the Z-axis, saves them as ``.npy``
arrays, and maintains a manifest CSV.

Usage::

    python -m medshield.data.slicer --data-dir data/raw --output-dir data/slices
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from medshield.data.loader import MODALITIES, MRIVolume, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slice Selection Strategies
# ---------------------------------------------------------------------------


class SliceStrategy(Enum):
    """Strategies for selecting 2D slices from a 3D volume."""

    TUMOR_ONLY = "tumor_only"
    CENTRAL = "central"
    UNIFORM = "uniform"
    ALL = "all"


def select_tumor_slices(seg_data: np.ndarray, min_tumor_pixels: int = 10) -> list[int]:
    """Return Z-indices of slices that contain at least *min_tumor_pixels*.

    Parameters
    ----------
    seg_data : np.ndarray
        3D segmentation mask (labels).
    min_tumor_pixels : int
        Minimum number of non-background (label > 0) pixels required.
    """
    tumor_pixels_per_slice = np.sum(seg_data > 0, axis=(0, 1))
    return np.where(tumor_pixels_per_slice >= min_tumor_pixels)[0].tolist()


def select_central_slices(depth: int, num_slices: int = 5) -> list[int]:
    """Return indices for *num_slices* contiguous slices around the center."""
    if num_slices >= depth:
        return list(range(depth))
    start = (depth - num_slices) // 2
    return list(range(start, start + num_slices))


def select_uniform_slices(depth: int, num_slices: int = 5) -> list[int]:
    """Return indices evenly spaced across the volume depth."""
    if num_slices >= depth:
        return list(range(depth))
    # Avoid the very first and very last slices if possible by using linspace
    indices = np.linspace(0, depth - 1, num=num_slices, dtype=int)
    return indices.tolist()


def select_slices(
    strategy: SliceStrategy,
    depth: int,
    seg_data: np.ndarray | None = None,
    num_slices: int = 5,
    min_tumor_pixels: int = 10,
) -> list[int]:
    """Select Z-indices based on the given *strategy*."""
    if strategy == SliceStrategy.TUMOR_ONLY:
        if seg_data is None:
            raise ValueError("Segmentation data required for TUMOR_ONLY strategy")
        return select_tumor_slices(seg_data, min_tumor_pixels)
    if strategy == SliceStrategy.CENTRAL:
        return select_central_slices(depth, num_slices)
    if strategy == SliceStrategy.UNIFORM:
        return select_uniform_slices(depth, num_slices)
    if strategy == SliceStrategy.ALL:
        return list(range(depth))
    raise ValueError(f"Unknown slice strategy: {strategy}")


# ---------------------------------------------------------------------------
# Slice Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedSlice:
    """A 2D slice extracted from a 3D patient volume.

    Attributes
    ----------
    patient_id : str
        Source patient identifier.
    slice_idx : int
        Z-index within the 3D volume.
    data : np.ndarray
        Multi-channel image array of shape ``(C, H, W)``.
    tumor_pixels : int
        Number of non-zero segmentation pixels in this slice.
    has_tumor : bool
        Whether this slice contains any tumor pixels.
    """

    patient_id: str
    slice_idx: int
    data: np.ndarray
    tumor_pixels: int
    has_tumor: bool


class SliceExtractor:
    """Extracts 2D slices from a loaded patient dictionary.

    Parameters
    ----------
    modalities : tuple[str, ...]
        The modalities to stack into channels (order matters).
    strategy : SliceStrategy
        The strategy for choosing which Z-slices to extract.
    num_slices : int
        Number of slices to extract (for CENTRAL and UNIFORM strategies).
    min_tumor_pixels : int
        Minimum tumor pixels (for TUMOR_ONLY strategy).
    """

    def __init__(
        self,
        modalities: tuple[str, ...] = MODALITIES,
        strategy: SliceStrategy = SliceStrategy.TUMOR_ONLY,
        num_slices: int = 5,
        min_tumor_pixels: int = 10,
    ) -> None:
        self.modalities = modalities
        self.strategy = strategy
        self.num_slices = num_slices
        self.min_tumor_pixels = min_tumor_pixels

    def extract(self, patient: dict[str, MRIVolume]) -> list[ExtractedSlice]:
        """Extract multi-channel 2D slices from the *patient* dictionary.

        Returns
        -------
        list[ExtractedSlice]
            A list of the extracted slices.
        """
        # Validate patient dict has all required modalities
        for mod in self.modalities:
            if mod not in patient:
                raise ValueError(f"Patient missing required modality: {mod}")

        # Assume all volumes have the same shape (verified by inspect module)
        base_vol = patient[self.modalities[0]]
        depth = base_vol.shape[2]
        patient_id = base_vol.patient_id

        seg_data = patient["seg"].data if "seg" in patient else None

        # Determine which slices to extract
        slice_indices = select_slices(
            self.strategy,
            depth=depth,
            seg_data=seg_data,
            num_slices=self.num_slices,
            min_tumor_pixels=self.min_tumor_pixels,
        )

        extracted = []
        for z in slice_indices:
            # Stack the 4 modalities into (C, H, W)
            channels = [patient[mod].data[:, :, z] for mod in self.modalities]
            stacked = np.stack(channels, axis=0)

            # Count tumor pixels if segmentation is available
            tumor_pixels = 0
            if seg_data is not None:
                tumor_pixels = int(np.sum(seg_data[:, :, z] > 0))

            extracted.append(
                ExtractedSlice(
                    patient_id=patient_id,
                    slice_idx=z,
                    data=stacked,
                    tumor_pixels=tumor_pixels,
                    has_tumor=tumor_pixels > 0,
                )
            )

        return extracted


# ---------------------------------------------------------------------------
# Dataset Pipeline
# ---------------------------------------------------------------------------


def save_spot_checks(slices: list[ExtractedSlice], output_dir: Path, num_samples: int = 10) -> None:
    """Save a visual representation of random extracted slices.

    Generates composite PNGs where the RGB channels map to T1ce, T2, and FLAIR.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Pick random samples
    if len(slices) > num_samples:
        samples = rng.choice(slices, size=num_samples, replace=False).tolist()
    else:
        samples = slices

    for sample in samples:
        # Assuming channels are (T1, T1ce, T2, FLAIR) -> index 1, 2, 3
        # Create a false-color RGB image
        if sample.data.shape[0] >= 4:
            r = sample.data[1, :, :]  # T1ce
            g = sample.data[3, :, :]  # FLAIR
            b = sample.data[2, :, :]  # T2
        else:
            # Fallback if fewer channels
            r = sample.data[0, :, :]
            g = sample.data[0, :, :]
            b = sample.data[0, :, :]

        # Simple min-max norm for visualization
        rgb = np.stack([r, g, b], axis=-1)
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        # Avoid division by zero
        denom = np.where(rgb_max - rgb_min == 0, 1.0, rgb_max - rgb_min)
        rgb = (rgb - rgb_min) / denom

        fig, ax = plt.subplots(figsize=(4, 4))
        # Transpose so it matches radiological convention (origin lower)
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
        ax.axis("off")
        title = (
            f"{sample.patient_id} - Slice {sample.slice_idx}\nTumor Pixels: {sample.tumor_pixels}"
        )
        ax.set_title(title, fontsize=9)

        out_path = output_dir / f"spotcheck_{sample.patient_id}_{sample.slice_idx}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def extract_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    strategy: SliceStrategy = SliceStrategy.TUMOR_ONLY,
    num_slices: int = 5,
) -> None:
    """Extract 2D slices from the full dataset and save to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = output_dir / "npy"
    slices_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"

    extractor = SliceExtractor(strategy=strategy, num_slices=num_slices)
    patient_generator = load_dataset(data_dir, include_seg=True)

    class_counts: Counter[str] = Counter()
    total_slices = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "patient_id", "slice_idx", "has_tumor", "tumor_pixels"])

        # Cache a few for spot checking
        spot_check_pool: list[ExtractedSlice] = []

        for patient in patient_generator:
            try:
                extracted = extractor.extract(patient)
            except Exception as exc:
                logger.error("Failed to extract slices for a patient: %s", exc)
                continue

            for ext in extracted:
                filename = f"{ext.patient_id}_slice{ext.slice_idx:03d}.npy"
                out_path = slices_dir / filename
                # Save as half precision or float32 depending on space. We'll use float32.
                np.save(out_path, ext.data)

                writer.writerow(
                    [
                        filename,
                        ext.patient_id,
                        ext.slice_idx,
                        ext.has_tumor,
                        ext.tumor_pixels,
                    ]
                )

                label_str = "tumor" if ext.has_tumor else "no-tumor"
                class_counts[label_str] += 1
                total_slices += 1

                if len(spot_check_pool) < 100:
                    spot_check_pool.append(ext)

    logger.info("Extraction complete. %d slices saved to %s", total_slices, slices_dir)
    logger.info("Manifest written to %s", manifest_path)
    logger.info("Class distribution: %s", dict(class_counts))

    if spot_check_pool:
        spot_check_dir = output_dir / "spot_checks"
        save_spot_checks(spot_check_pool, spot_check_dir)
        logger.info("Visual spot checks saved to %s", spot_check_dir)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the slice extraction pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract 2D slices from BraTS NIfTI volumes.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing the BraTS patient folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/slices"),
        help="Directory to write extracted slices and manifest.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="tumor_only",
        choices=[s.value for s in SliceStrategy],
        help="Strategy for selecting slices (default: tumor_only).",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="Number of slices to extract (for central/uniform strategies).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = SliceStrategy(args.strategy)

    extract_dataset(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        strategy=strategy,
        num_slices=args.num_slices,
    )


if __name__ == "__main__":
    main()
