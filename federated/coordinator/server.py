"""
federated/coordinator/server.py

Central Flower FL coordinator server.
Responsibilities:
  - Loads initial global ArcFace model weights and distributes to org nodes
  - Runs N federation rounds (FedAvg + optional DP)
  - Saves aggregated model after each round via ModelManager
  - Exposes federation controls (start, stop) via CLI

Usage:
    python -m federated.coordinator.server [--rounds 10] [--address 0.0.0.0:8080]

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server import ServerConfig

# Add project root to path so cross-module imports work
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.coordinator.strategy import FedAvgWithDP
from federated.coordinator.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coordinator.server")


# ──────────────────────────────────────────────────────────────────
# Initial global model weights
# ──────────────────────────────────────────────────────────────────

def load_initial_weights(weights_path: Optional[str] = None) -> List[np.ndarray]:
    """
    Load initial global model weights.

    Priority:
      1. Load from weights_path (.npz file) if provided and exists.
      2. Load previously saved global model from ModelManager.
      3. Fall back to random initialisation (for testing without pretrained weights).

    In production, weights_path should point to a pretrained ArcFace checkpoint
    exported as a .npz via vision/arcface_model.py.
    """
    if weights_path and os.path.exists(weights_path):
        logger.info("Loading initial weights from %s", weights_path)
        archive = np.load(weights_path, allow_pickle=False)
        return [archive[k] for k in sorted(archive.files)]

    # Try loading from ModelManager (resuming a previous run)
    mm = ModelManager()
    existing = mm.load_latest_weights()
    if existing is not None:
        logger.info("Resuming from latest saved global model (v%d).", mm.current_version)
        return existing

    # Random initialisation — only valid for smoke-testing
    logger.warning(
        "No pretrained weights found. Using random initialisation. "
        "Set --weights to a pretrained ArcFace .npz for real training."
    )
    # Minimal dummy weights that match a small embedding model for testing
    return [np.zeros((512,), dtype=np.float32)]


# ──────────────────────────────────────────────────────────────────
# Fit & Evaluate config functions (sent to clients each round)
# ──────────────────────────────────────────────────────────────────

def fit_config(server_round: int) -> Dict:
    """Config dict sent to each client's fit() call."""
    return {
        "server_round": server_round,
        "local_epochs": 1,          # Increase for later rounds if needed
        "learning_rate": 1e-4,
        "use_dp": True,
        "noise_multiplier": 1.1,    # Opacus DP noise (client-side)
        "max_grad_norm": 1.0,
    }


def eval_config(server_round: int) -> Dict:
    """Config dict sent to each client's evaluate() call."""
    return {"server_round": server_round}


# ──────────────────────────────────────────────────────────────────
# Post-fit callback — save model after each round
# ──────────────────────────────────────────────────────────────────

def make_fit_metrics_aggregation_fn(model_manager: ModelManager):
    """
    Returns a function that aggregates fit metrics and saves the model.
    Passed to FedAvg as fit_metrics_aggregation_fn.
    """
    def fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        total = sum(n for n, _ in metrics)
        avg_loss = sum(n * m.get("loss", 0.0) for n, m in metrics) / (total or 1)
        return {"avg_train_loss": avg_loss}
    return fn


# ──────────────────────────────────────────────────────────────────
# Main server entry point
# ──────────────────────────────────────────────────────────────────

def start_coordinator(
    num_rounds: int = 10,
    server_address: str = "0.0.0.0:8080",
    min_clients: int = 2,
    noise_multiplier: float = 0.0,   # server-side DP (0 = off; Opacus handles client-side)
    weights_path: Optional[str] = None,
    model_dir: str = "models/global",
):
    """
    Start the Flower federated learning coordinator server.

    Args:
        num_rounds:       Number of federation rounds.
        server_address:   gRPC address to listen on.
        min_clients:      Minimum org nodes required to start a round.
        noise_multiplier: Server-side DP noise (0 disables server-side noise).
        weights_path:     Path to initial pretrained ArcFace weights (.npz).
        model_dir:        Directory to save global model versions.
    """
    model_manager = ModelManager(model_dir=model_dir)
    initial_weights = load_initial_weights(weights_path)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = FedAvgWithDP(
        # Flower FedAvg base args
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=make_fit_metrics_aggregation_fn(model_manager),
        # Our extensions
        noise_multiplier=noise_multiplier,
        model_manager=model_manager,
    )

    config = ServerConfig(num_rounds=num_rounds)

    logger.info("=" * 60)
    logger.info("Starting Federated Coordinator")
    logger.info("  Address      : %s", server_address)
    logger.info("  Rounds       : %d", num_rounds)
    logger.info("  Min clients  : %d", min_clients)
    logger.info("  Server DP    : %s (σ=%.3f)", noise_multiplier > 0, noise_multiplier)
    logger.info("  Model dir    : %s", model_dir)
    logger.info("=" * 60)

    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Learning Coordinator — Missing Person Identification"
    )
    parser.add_argument("--rounds",   type=int,   default=10,              help="Number of federation rounds")
    parser.add_argument("--address",  type=str,   default="0.0.0.0:8080", help="gRPC server address")
    parser.add_argument("--clients",  type=int,   default=2,               help="Minimum org nodes required")
    parser.add_argument("--dp-noise", type=float, default=0.0,             help="Server-side DP noise multiplier")
    parser.add_argument("--weights",  type=str,   default=None,            help="Path to initial .npz weights")
    parser.add_argument("--model-dir",type=str,   default="models/global", help="Directory to save global models")
    args = parser.parse_args()

    start_coordinator(
        num_rounds=args.rounds,
        server_address=args.address,
        min_clients=args.clients,
        noise_multiplier=args.dp_noise,
        weights_path=args.weights,
        model_dir=args.model_dir,
    )
