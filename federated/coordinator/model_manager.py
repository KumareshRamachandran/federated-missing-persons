"""
federated/coordinator/model_manager.py

Manages the global ArcFace model across federation rounds.
Handles: versioning, persistence to disk, accuracy history tracking.

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import os
import json
import time
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages global model versioning and persistence across federation rounds.

    Stores each round's aggregated weights as a .npz file and tracks
    accuracy metrics in a JSON history file.
    """

    HISTORY_FILE = "history.json"

    def __init__(self, model_dir: str = "models/global"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        history_path = os.path.join(model_dir, self.HISTORY_FILE)
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.history: list = json.load(f)
            self.current_version: int = self.history[-1]["version"] if self.history else 0
        else:
            self.history: list = []
            self.current_version: int = 0

        logger.info(
            "ModelManager initialised. Dir=%s, current_version=%d",
            model_dir, self.current_version
        )

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def save_model(
        self,
        weights: list,          # list of np.ndarray — aggregated model weights
        accuracy: float,
        server_round: int,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Persist a new version of the global model to disk.

        Args:
            weights:      Aggregated weight arrays from FedAvg.
            accuracy:     Evaluation accuracy for this round (0.0 – 1.0).
            server_round: Current federation round number.
            metadata:     Optional extra info (epsilon, num_clients, etc.).

        Returns:
            New version number.
        """
        self.current_version += 1
        fname = f"global_v{self.current_version:04d}_round{server_round}.npz"
        fpath = os.path.join(self.model_dir, fname)

        # Save weight arrays as compressed NumPy archive
        np.savez_compressed(fpath, *weights)

        record = {
            "version": self.current_version,
            "round": server_round,
            "accuracy": round(accuracy, 6),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file": fname,
            **(metadata or {}),
        }
        self.history.append(record)
        self._persist_history()

        logger.info(
            "Saved global model v%d | round=%d | accuracy=%.4f | file=%s",
            self.current_version, server_round, accuracy, fname
        )
        return self.current_version

    def load_latest_weights(self) -> Optional[list]:
        """
        Load the most recently saved global model weights.

        Returns:
            List of np.ndarray weight arrays, or None if no model saved yet.
        """
        if not self.history:
            logger.warning("No saved model found in %s", self.model_dir)
            return None

        latest = self.history[-1]
        fpath = os.path.join(self.model_dir, latest["file"])

        if not os.path.exists(fpath):
            logger.error("Model file missing: %s", fpath)
            return None

        archive = np.load(fpath, allow_pickle=False)
        weights = [archive[k] for k in sorted(archive.files)]
        logger.info("Loaded model v%d from %s", latest["version"], fpath)
        return weights

    def load_version(self, version: int) -> Optional[list]:
        """Load a specific model version by version number."""
        record = next((r for r in self.history if r["version"] == version), None)
        if record is None:
            logger.error("Version %d not found in history.", version)
            return None

        fpath = os.path.join(self.model_dir, record["file"])
        archive = np.load(fpath, allow_pickle=False)
        return [archive[k] for k in sorted(archive.files)]

    def get_accuracy_history(self) -> list:
        """
        Return accuracy metrics for every federation round.

        Returns:
            List of dicts: [{version, round, accuracy, timestamp, ...}]
        """
        return self.history

    def get_best_accuracy(self) -> float:
        """Return the highest accuracy achieved across all rounds."""
        if not self.history:
            return 0.0
        return max(r["accuracy"] for r in self.history)

    def get_latest_record(self) -> Optional[dict]:
        """Return the metadata dict for the most recent round."""
        return self.history[-1] if self.history else None

    def history_update_accuracy(self, server_round: int, accuracy: float):
        """
        Update the accuracy field for a given round in history
        (called from strategy.aggregate_evaluate where we don't have weights yet).
        """
        for record in reversed(self.history):
            if record["round"] == server_round:
                record["accuracy"] = round(accuracy, 6)
                self._persist_history()
                return
        # If no record yet for this round, create a placeholder
        self.history.append({
            "version": self.current_version,
            "round": server_round,
            "accuracy": round(accuracy, 6),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file": None,
        })
        self._persist_history()

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _persist_history(self):
        """Write history list to JSON on disk."""
        path = os.path.join(self.model_dir, self.HISTORY_FILE)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
