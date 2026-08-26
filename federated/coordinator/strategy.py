"""
coordinator/strategy.py

Custom FedAvg aggregation strategy with optional Differential Privacy noise injection.
Extends the default Flower FedAvg strategy.
"""

import flwr as fl
from flwr.server.strategy import FedAvg


class FedAvgWithDP(FedAvg):
    """
    Custom strategy: Weighted FedAvg + Differential Privacy on aggregated weights.
    """

    def __init__(self, noise_multiplier: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier

    def aggregate_fit(self, server_round, results, failures):
        # TODO: Call super().aggregate_fit() to get base aggregated weights
        # TODO: If noise_multiplier > 0, inject Gaussian noise for DP
        # TODO: Return noisy aggregated weights
        pass

    def aggregate_evaluate(self, server_round, results, failures):
        # TODO: Aggregate evaluation metrics (accuracy, loss) across clients
        pass
