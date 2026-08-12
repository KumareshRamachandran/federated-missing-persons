"""
coordinator/model_manager.py

Tracks global model versions, accuracy metrics per round,
and handles model serialization / deserialization.
"""

import os
import torch


class ModelManager:
    """Manages global model versioning and persistence."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.current_version = 0
        self.history = []  # List of {"version": int, "accuracy": float, "round": int}
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, model_state_dict: dict, accuracy: float):
        """Save a new version of the global model."""
        # TODO: Increment version, save state_dict to disk
        # TODO: Append to self.history
        pass

    def load_latest_model(self) -> dict:
        """Load the latest global model state dict."""
        # TODO: Load from disk by version number
        pass

    def get_accuracy_history(self) -> list:
        """Return accuracy over all federation rounds."""
        return self.history
