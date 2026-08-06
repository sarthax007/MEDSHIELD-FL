"""Experiment logging utilities.

Task 29 - Implement experiment logging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs hyperparameter configurations and metrics for a training run."""

    def __init__(self, log_dir: str, run_id: str):
        """Initialise the experiment logger.

        Parameters
        ----------
        log_dir : str
            The base directory for all logs.
        run_id : str
            A unique identifier for this run.
        """
        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.hyperparameters_file = self.run_dir / "hyperparameters.json"
        logger.info(f"Initialized ExperimentLogger at {self.run_dir}")

    def log_hyperparameters(
        self, config: ModelConfig, extra: Optional[dict[str, Any]] = None
    ) -> None:
        """Log the model configuration and any extra hyperparameters.

        Parameters
        ----------
        config : ModelConfig
            The model configuration.
        extra : Optional[Dict[str, Any]], default=None
            Additional hyperparameters to log.
        """
        params = config.to_dict()
        if extra:
            params.update(extra)

        with open(self.hyperparameters_file, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Logged hyperparameters to {self.hyperparameters_file}")

    def log_metrics(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log metrics for a specific epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        metrics : Dict[str, Any]
            The metrics to log.
        """
        record = {"epoch": epoch}
        record.update(metrics)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
