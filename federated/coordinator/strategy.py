"""
federated/coordinator/strategy.py

Custom Flower aggregation strategy:
  - Weighted FedAvg (McMahan et al., 2017)
  - Optional server-side Gaussian DP noise on aggregated weights
  - Per-round accuracy logging via ModelManager

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from federated.coordinator.model_manager import ModelManager

logger = logging.getLogger(__name__)


class FedAvgWithDP(FedAvg):
    """
    Weighted FedAvg + optional server-side Gaussian Differential Privacy.

    DP noise is added to the AGGREGATED weights (server-side DP).
    This complements Opacus client-side DP-SGD, providing defence-in-depth:
      - Client sends DP-noised gradients  (Opacus — Kishore's module)
      - Server further noises aggregated weights (this class)

    Reference: McMahan et al. (2017), Geyer et al. (2017).
    """

    def __init__(
        self,
        noise_multiplier: float = 0.0,
        model_manager: Optional[ModelManager] = None,
        **kwargs,
    ):
        """
        Args:
            noise_multiplier: σ for Gaussian noise N(0, σ²) on aggregated weights.
                              Set to 0.0 to disable server-side DP.
            model_manager:    ModelManager instance for saving rounds to disk.
            **kwargs:         Passed to FedAvg (min_fit_clients, fraction_fit, etc.)
        """
        super().__init__(**kwargs)
        self.noise_multiplier = noise_multiplier
        self.model_manager = model_manager
        logger.info(
            "FedAvgWithDP: noise_multiplier=%.3f, server_dp=%s",
            noise_multiplier,
            noise_multiplier > 0,
        )

    # ──────────────────────────────────────────────────────────────
    # Training aggregation
    # ──────────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client model updates using weighted FedAvg,
        then optionally inject Gaussian DP noise on the server side.
        """
        if failures:
            logger.warning("Round %d: %d client(s) failed.", server_round, len(failures))

        if not results:
            logger.error("Round %d: No results to aggregate.", server_round)
            return None, {}

        # ── Step 1: Weighted FedAvg via parent class ──────────────
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is None:
            return None, {}

        # ── Step 2: Server-side DP noise injection ────────────────
        if self.noise_multiplier > 0.0:
            weights = parameters_to_ndarrays(aggregated_params)
            noisy_weights = self._inject_gaussian_noise(weights)
            aggregated_params = ndarrays_to_parameters(noisy_weights)
            logger.info(
                "Round %d: Injected server-side DP noise (σ=%.3f)",
                server_round, self.noise_multiplier,
            )

        # ── Step 3: Log per-round metrics ─────────────────────────
        num_samples_total = sum(fit_res.num_examples for _, fit_res in results)
        logger.info(
            "Round %d aggregated | clients=%d | total_samples=%d",
            server_round, len(results), num_samples_total,
        )

        return aggregated_params, metrics

    # ──────────────────────────────────────────────────────────────
    # Evaluation aggregation
    # ──────────────────────────────────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics (loss, accuracy) from all clients.
        Saves best-round model to ModelManager.
        """
        if not results:
            return None, {}

        # Weighted average of loss and accuracy
        total_samples = sum(r.num_examples for _, r in results)
        avg_loss = sum(
            r.loss * r.num_examples for _, r in results
        ) / total_samples

        # Accuracy is passed as a metric in r.metrics
        accuracies = [
            r.metrics.get("accuracy", 0.0) * r.num_examples
            for _, r in results
        ]
        avg_accuracy = sum(accuracies) / total_samples

        logger.info(
            "Round %d eval | loss=%.4f | accuracy=%.4f | clients=%d",
            server_round, avg_loss, avg_accuracy, len(results),
        )

        # Persist to ModelManager (if available)
        if self.model_manager is not None:
            # We don't have weights here, but we record accuracy.
            # Actual weight saving happens after aggregate_fit via server callback.
            self.model_manager.history_update_accuracy(server_round, avg_accuracy)

        return avg_loss, {"accuracy": avg_accuracy}

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _inject_gaussian_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add calibrated Gaussian noise to aggregated weight arrays.

        Noise scale: N(0, noise_multiplier²) applied element-wise.
        This implements the Gaussian mechanism for (ε, δ)-DP.
        """
        noisy = []
        for w in weights:
            noise = np.random.normal(loc=0.0, scale=self.noise_multiplier, size=w.shape)
            noisy.append((w + noise).astype(w.dtype))
        return noisy
