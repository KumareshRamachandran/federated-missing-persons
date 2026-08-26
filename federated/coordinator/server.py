"""
coordinator/server.py

The central Flower server (coordinator).
Responsibilities:
  - Initialize and distribute the global ArcFace model to all org nodes
  - Aggregate model weight updates using FedAvg
  - Route search queries to org nodes and collect Match/No-Match results
  - Track model version and accuracy per federation round
"""

import flwr as fl
from coordinator.strategy import FedAvgWithDP


def start_coordinator(num_rounds: int = 10, server_address: str = "0.0.0.0:8080"):
    """Start the Flower federated learning coordinator server."""
    # TODO: Initialize global model
    # TODO: Define fit/eval config functions
    # TODO: Start fl.server.start_server(...)
    pass


if __name__ == "__main__":
    start_coordinator()
