"""Model configuration and serialization.

Task 23 — Implement a model configuration and registry.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class ModelConfig:
    """Configuration for the Vision Transformer model."""

    model_name: str = "vit_base_patch16_224"
    num_classes: int = 2
    pretrained: bool = True
    input_size: int = 224
    drop_rate: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    mixed_precision: bool = False

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialize config from dictionary."""
        return cls(**data)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
