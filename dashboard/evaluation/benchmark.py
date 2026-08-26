"""
evaluation/benchmark.py

Runs a full evaluation benchmark comparing:
  1. Centralized baseline: All gallery embeddings merged at coordinator
  2. Federated (our system): Distributed matching with binary result aggregation

Outputs a comparison table and saves results to evaluation/results/
"""

import json
import os


def run_centralized_benchmark(query_dir: str, gallery_dir: str) -> dict:
    """
    Simulate centralized matching: all gallery embeddings at one node.

    Returns:
        {"rank1_accuracy": float, "avg_latency_ms": float}
    """
    # TODO: Load all gallery embeddings
    # TODO: For each query, find best match across all galleries
    # TODO: Compute rank1 accuracy and average query latency
    pass


def run_federated_benchmark(query_dir: str, node_dirs: list, threshold: float = 0.6) -> dict:
    """
    Simulate federated matching: each node matches locally, returns binary result.

    Returns:
        {"rank1_accuracy": float, "avg_latency_ms": float, "privacy_preserved": True}
    """
    # TODO: For each query, send embedding to each node's LocalMatcher
    # TODO: Aggregate Match/No-Match results
    # TODO: Compute rank1 accuracy and average latency
    pass


def compare_and_report(results_centralized: dict, results_federated: dict, output_path: str):
    """Print and save comparison table."""
    # TODO: Print side-by-side comparison
    # TODO: Save as JSON to output_path
    pass


if __name__ == "__main__":
    os.makedirs("evaluation/results", exist_ok=True)
    c = run_centralized_benchmark("data/query_set", "data/nodes")
    f = run_federated_benchmark("data/query_set", ["data/nodes/node_police", "data/nodes/node_hospital", "data/nodes/node_ngo"])
    compare_and_report(c, f, "evaluation/results/comparison.json")
