"""Dataset loading, preprocessing, labelling and hospital partitioning (Tasks 9-20)."""

from .augmentation import get_eval_transforms, get_train_transforms, visualize_augmentations
from .dataset import BraTS2DSliceDataset, create_dataloaders
from .labels import TumorClass, confirm_mapping, map_raw_label
from .loader import MRIVolume, VolumeLoadError, load_patient, load_volume
from .partitioning import partition_by_patient
from .preprocess import preprocess_slice
from .slicer import SliceExtractor, SliceStrategy
from .splitting import create_hospital_splits, process_all_clients
from .validate import generate_report, validate_hospital_splits

__all__ = [
    "MRIVolume",
    "VolumeLoadError",
    "load_patient",
    "load_volume",
    "SliceExtractor",
    "SliceStrategy",
    "preprocess_slice",
    "TumorClass",
    "map_raw_label",
    "confirm_mapping",
    "get_train_transforms",
    "get_eval_transforms",
    "visualize_augmentations",
    "partition_by_patient",
    "BraTS2DSliceDataset",
    "create_dataloaders",
    "create_hospital_splits",
    "process_all_clients",
    "validate_hospital_splits",
    "generate_report",
]
