"""Tests for experiment logging utilities.

Task 29 - Implement experiment logging.
"""

import json
import os
from tempfile import TemporaryDirectory

from medshield.model import ExperimentLogger, ModelConfig


def test_experiment_logger_creates_files():
    """Test that the logger creates the expected run directory and files."""
    config = ModelConfig(model_name="test_vit", num_classes=3)
    
    with TemporaryDirectory() as temp_dir:
        logger = ExperimentLogger(log_dir=temp_dir, run_id="run_01")
        
        # Log hyperparameters
        logger.log_hyperparameters(config, extra={"batch_size": 32})
        
        # Log metrics across two epochs
        logger.log_metrics(epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4})
        logger.log_metrics(epoch=2, metrics={"train_loss": 0.3, "val_loss": 0.2})
        
        run_path = os.path.join(temp_dir, "run_01")
        assert os.path.isdir(run_path)
        
        params_path = os.path.join(run_path, "hyperparameters.json")
        metrics_path = os.path.join(run_path, "metrics.jsonl")
        
        assert os.path.exists(params_path)
        assert os.path.exists(metrics_path)
        
        # Verify hyperparameters content
        with open(params_path, "r") as f:
            params_data = json.load(f)
        assert params_data["model_name"] == "test_vit"
        assert params_data["num_classes"] == 3
        assert params_data["batch_size"] == 32
        
        # Verify metrics content
        with open(metrics_path, "r") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        m1 = json.loads(lines[0])
        assert m1["epoch"] == 1
        assert m1["train_loss"] == 0.5
        
        m2 = json.loads(lines[1])
        assert m2["epoch"] == 2
        assert m2["val_loss"] == 0.2
